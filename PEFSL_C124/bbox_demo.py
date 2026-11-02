# -*- coding: utf-8 -*-
"""
bbox_demo.py — PEFSL 框选式小样本自学习与识别演示
==================================================

A 模式（学习）：
    鼠标框选摄像头画面中的物体 → 输入类别标签 → 程序按 PEFSL 算法
    自动采集该框内多帧特征（每帧含中心裁剪增强，抑制背景突出物体），
    在线完成“自学习”（注册即学习，无需梯度训练）。
    同一标签可多次框选（不同位置/角度），样本自动累积。
    注册时自动检查：框太大/物体太小/两类特征过于相似都会给出提示。

B 模式（识别）：
    默认自动定位并框出画面中的已学物体（框上标注类别与相似度）；
    未定位到物体时对整幅画面持续识别（中央大字显示最佳匹配或"无匹配"）；
    框选物体 → 只识别框内区域；按 s 可随时关闭/开启自动定位。
    每个已学类别旁实时显示相似度条，直观看到模型的判断依据。
    显示的相似度结合最高样本与Top-K平均，不是正确概率。
    新登记类别通过完整参考对齐、几何匹配等证据核验当前身份；
    颜色、背景与用户负样本仍可拒绝。无法辨识的纹理、光照、遮挡和移动
    仍可能造成漏检，不能用高相似度保证未知物体识别正确。

PEFSL 算法流程（与 main.py 原始演示完全一致）：
    1. ResNet-9 骨干网络提取图像特征（80 维）
    2. feature_preprocess：单位球归一化；旧兼容类别减去固定基准均值
    3. NCM（最近类均值）/ KNN 分类器在线分类
    4. 指数滑动平均（EMA）平滑逐帧预测概率
    其中 2/3/4 直接复用 few_shot_model 模块的实现。

用法（在 PEFSL-main 目录下）：
    python bbox_demo.py
    python bbox_demo.py --camera-id 0 --camera-resolution 640x480 --output-resolution 800x600
    python bbox_demo.py --classifier-type knn --shots-per-box 20

按键：
    a / b     切换 A 模式（学习）/ B 模式（识别）
    s         B 模式下开关自动扫描
    c         清除当前识别框
    r         清空全部已学习数据
    w / l     保存 / 加载已学习数据（保存到 --save-path）
    q / ESC   退出
    鼠标左键拖拽：框选物体

自检（不依赖摄像头/GUI，验证学习-识别全流程）：
    python bbox_demo.py --self-test
"""

import argparse
import os
import sys
import time
import copy
from contextlib import contextmanager
import threading
import zipfile
import uuid
from collections import deque, OrderedDict

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# PEFSL 原始模块：骨干网络加载器 + 小样本分类核心函数
from bbox_storage import save_archive_snapshot
from backbone_loader.backbone_loader import get_model
from few_shot_model.few_shot_model import feature_preprocess, ncm, knn
from bbox_localization import (fit_bright_component, bright_component_candidates,
                               endpoint_boundary_support, bright_body_contour,
                               transparent_extension_density, shell_extension_support,
                               competing_shell_evidence,
                               anchored_shell_candidates, measured_tapered_boundary,
                               reflected_shell_candidates, measured_narrow_tip)
from bbox_colour import chromatic_model, colour_component_candidates
from bbox_generic import (add_reference_view, pack_reference_views,
                          unpack_reference_views, generic_colour_supported, generic_colour_query,
                          scene_reference_features, project_reference_candidates,
                          template_reference_candidates, object_region_candidates)
from bbox_dense import (dense_reference_features, match_dense_reference, recheck_dense_hint, indexed_registration_patterns,
                        spatial_reference_features, match_spatial_reference, match_fine_spatial_reference,
                        spatial_reference_choices, spatial_scene_candidates, recheck_spatial_hint,
                        recheck_learned_points, IDENTITY_INCOMPLETE)
from bbox_learning import RegistrationCapture
from bbox_search import select_diverse, local_variants, registration_correlations
from bbox_tracking import (seed_points, track_points, shift_box, describe_appearance,
                           relocate_template, reference_sharpness, recovery_region,LocalCorrelationTracker, APPEARANCE_SIZE)
from bbox_runtime import LatestFrameCamera, SingleFlightWorker, WindowsRealtimePolicy, DiagnosticWriter, validate_search_result
from bbox_diagnostics import identity_attempt, diagnostic_context
from bbox_identity import (registration_patterns, match_registration_pattern,
                           registered_identity_supported,match_colour_registration_pattern,
                           colour_registration_features,match_colour_geometry,recheck_registration_hint)

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

WIN_NAME = "PEFSL-BBox-Demo"

# 与 main.py preprocess() 相同的归一化常数（BGR 顺序）
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# 各类别框选颜色（BGR，循环取用）
CLASS_COLORS = [
    (0, 0, 255), (0, 200, 0), (255, 60, 0), (0, 140, 255),
    (255, 0, 200), (200, 0, 255), (0, 255, 255), (255, 255, 0),
]
GREEN_BOX = (0, 220, 0)
GRAY_BOX = (160, 160, 160)
WHITE = (255, 255, 255)
YELLOW = (0, 230, 230)
RED = (0, 0, 255)
HEAD_BG = (25, 25, 25)

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\Deng.ttf",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
    r"C:\Windows\Fonts\arial.ttf",
]

# 界面文案： (中文, 英文)
TEXTS = {
    "title":       ("PEFSL 自学习识别演示", "PEFSL Learn & Recognize"),
    "mode_a":      ("A 模式 · 学习", "Mode A - Learn"),
    "mode_b":      ("B 模式 · 识别", "Mode B - Recognize"),
    "hint_a":      ("贴边框选物体(尽量别框进手/桌面, 可把物体放桌上框) → 输入标签 → 自动采集",
                    "Draw a TIGHT box on the object (avoid hand/desk in box), then type a label"),
    "hint_b":      ("B 模式: 自动定位并框出已学物体; 也可举近/框选识别; s:定位开关", "Auto-locate & box objects / bring close / drag a box / s:locator"),
    "prompt_label": ("输入类别标签:", "Enter class label:"),
    "prompt_hint": ("回车确认(留空自动命名) · ESC取消 · 退格删除 · 英文输入法", "Enter=OK(blank=auto) ESC=cancel BKSP=del"),
    "collecting":  ("采集中：缓慢翻转/转动物体，采集多个角度 ...", "Collecting: slowly rotate object for more angles ..."),
    "no_data_b":   ("尚未学习任何类别，请先在 A 模式框选学习", "No class learned yet, learn in Mode A first"),
    "box_small":   ("框太小，请框大一些", "Box too small, draw a larger one"),
    "saved":       ("已保存学习数据", "Learned data saved"),
    "loaded":      ("已加载学习数据", "Learned data loaded"),
    "no_save":     ("未找到已保存的数据文件", "No saved data file found"),
    "nothing_save": ("无数据可保存", "Nothing to save"),
    "reset":       ("已清空全部学习数据", "All learned data cleared"),
    "cam_err":     ("摄像头读取失败，尝试恢复中 ...", "Camera read failed, recovering ..."),
    "cam_ok":      ("摄像头已恢复", "Camera recovered"),
    "scan_on":     ("物体定位：开（自动框出识别到的物体）", "Locator: ON (auto-box objects)"),
    "scan_off":    ("物体定位：关（仅整幅画面识别，不画框）", "Locator: OFF (whole-frame only, no box)"),
    "box_cleared": ("已清除识别框", "Recognition box cleared"),
    "no_match":    ("无匹配", "no match"),
    "reg_list":    ("已学类别", "Classes"),
    "fr":          ("帧", "fr"),
    "collect_n":   ("采集帧数: %d (+/- 调整)", "Frames/box: %d (+/- to adjust)"),
    "shots_set":   ("每次框选采集帧数已设为 %d", "Frames per box set to %d"),
    "hint_keys_a": ("A:学习  B:识别  拖拽框选  k:背景采样  +/-:帧数  r:清空  w:保存  l:加载  q:退出", "A:learn B:recog drag=box k:bg-sample +/-:frames r:clear w:save l:load q:quit"),
    "hint_keys_b": ("A:学习  B:识别  s:定位  k:背景采样  n:标记负样本  d:存诊断  +/-:帧数  c:清框  r:清空  w:保存  l:加载  q:退出", "A:learn B:recog s:locator k:bg-sample n:neg d:save-diag +/-:frames c:clear r:reset w:save l:load q:quit"),
    "unknown":     ("未识别到已学习物体", "No learned object found"),
    "motion_hold": ("移动跟踪中·待复核", "Tracking motion; awaiting verification"),
}


# ----------------------------------------------------------------------------
# 基础工具
# ----------------------------------------------------------------------------
def parse_wh(s):
    """'640x480' -> (640, 480)"""
    w, h = s.lower().split("x")
    return (int(w), int(h))


def preprocess_batch(imgs):
    """uint8 BGR (N,h,w,3) -> float32 归一化 (N,h,w,3)。与 main.py 的 preprocess 一致。"""
    x = imgs.astype(np.float32) / 255.0
    return x - MEAN / STD


_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
_ILLUM_STATE = threading.local()


def illum_norm(img):
    """暗光增强 + 光照归一化：对 Lab 的 L 通道做较强 CLAHE(更局部、更高
    对比度增益), 提升暗部细节——光线暗时特征对比度低、噪声大, 是漏检/
    误检的重要外部因素。注册与识别走同一条路径, 保证特征空间一致。

    CLAHE 只提升**局部**对比度, 整幅都偏暗时全局亮度仍然很低(暗部噪声
    被一起放大, 特征信噪比上不去)。所以再按整体亮度做一次自动增益: 越暗
    增益越强(封顶 1.6 倍, 避免把噪声拉满), 亮度正常时几乎不动作。
    这条增益对注册/识别两侧都是同一函数、同一判据, 不会破坏一致性。"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    # OpenCV CLAHE has working buffers; use one instance per search/UI thread.
    if not hasattr(_ILLUM_STATE,"clahe"):
        _ILLUM_STATE.clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(4,4))
    l = _ILLUM_STATE.clahe.apply(l)
    lm = float(l.mean())
    if lm < 100.0:
        gain = min(1.6, 100.0 / max(lm, 55.0))
        if gain > 1.02:
            l = cv2.convertScaleAbs(l, alpha=gain, beta=0)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def crop_resize(frame, box, size):
    """按 box=(x1,y1,x2,y2) 裁剪并缩放到 size×size（骨干网络输入）。size 可为 int 或 (s,s)。"""
    size = int(size[0] if isinstance(size, (tuple, list)) else size)
    x1, y1, x2, y2 = box
    crop = frame[y1:y2, x1:x2]
    assert crop.size > 0, f"invalid box {box}"
    interp = cv2.INTER_AREA if max(crop.shape[:2]) > size else cv2.INTER_LINEAR
    return illum_norm(cv2.resize(crop, (size, size), interpolation=interp))


def norm_box(p1, p2, w, h):
    """鼠标两个端点 -> 规范化 (x1,y1,x2,y2)，并裁剪到画面内。"""
    x1, x2 = sorted((max(0, min(p1[0], w - 1)), max(0, min(p2[0], w - 1))))
    y1, y2 = sorted((max(0, min(p1[1], h - 1)), max(0, min(p2[1], h - 1))))
    return (int(x1), int(y1), int(x2) + 1, int(y2) + 1)


def _iou(b1, b2):
    """两个 (x1,y1,x2,y2) 框的交并比——多目标重叠判定/去重用"""
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / float(a1 + a2 - inter)


def _same_obj(b1, b2, th=0.75):
    """两个框是否指向同一物体（跨类去重用）。
    用 交集/较小框面积 而非 IoU：IoU 在'大框套小框'（一个物体被两类
    同时咬住、其中一类框明显偏大）时远低于 0.5，同一物体会被重复框出；
    包含率对小框完全在大框内的情形 ≈1.0，能可靠识别同一物体。"""
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return False
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / float(min(a1, a2)) > th


def build_scan_boxes(w, h, side, cap, ar=1.0):
    """按类别宽高比覆盖画面；side 表示长边，默认保持方形兼容。"""
    side = int(max(16, side))
    if ar >= 1.0:
        sw, sh = side, max(16, int(round(side / ar)))
    else:
        sw, sh = max(16, int(round(side * ar))), side
    sw, sh = min(sw, w), min(sh, h)
    step = max(1, int(round(min(sw, sh) * 0.6)))
    while True:
        xs = list(range(0, max(w - sw, 0) + 1, step))
        ys = list(range(0, max(h - sh, 0) + 1, step))
        if xs and xs[-1] != w - sw:
            xs.append(w - sw)
        if ys and ys[-1] != h - sh:
            ys.append(h - sh)
        if len(xs) * len(ys) <= cap or step >= max(w, h):
            break
        step = max(step + 1, int(step * 1.35))
    return [(x, y, x + sw, y + sh) for y in ys for x in xs]


# ----------------------------------------------------------------------------
# 中文文字渲染（PIL，支持中英文；无字体时回退 cv2 英文）
# ----------------------------------------------------------------------------
class TextRenderer:
    def __init__(self, lang_pref="auto"):
        self.font_path = None
        for p in FONT_CANDIDATES:
            if os.path.exists(p):
                self.font_path = p
                break
        self.use_pil = HAS_PIL and self.font_path is not None
        if lang_pref == "auto":
            self.lang = "zh" if self.use_pil else "en"
        else:
            self.lang = lang_pref
            if lang_pref == "zh" and not self.use_pil:
                print("[warn] 未找到中文字体(Pillow)，界面将使用英文")
                self.lang = "en"
        self._fonts = {}
        self._text_tiles = OrderedDict()

    def t(self, key):
        return TEXTS[key][0 if self.lang == "zh" else 1]

    def text_width(self, text, px):
        if self.use_pil:
            return int(self._font(px).getlength(text))
        return int(len(text) * px * 0.6)

    def _font(self, px):
        px = max(10, int(px))
        if px not in self._fonts:
            self._fonts[px] = ImageFont.truetype(self.font_path, px)
        return self._fonts[px]

    def render(self, frame, ops):
        """把 [(text,x,y,px,bgr), ...] 渲染到 frame 上"""
        if not ops:
            return
        if self.use_pil:
            for text, x, y, px, bgr in ops:
                key=(text,int(px),tuple(bgr))
                tile=self._text_tiles.get(key)
                if tile is None:
                    font=self._font(px)
                    stroke=max(1,int(px)//18)
                    lo_x,lo_y,hi_x,hi_y=font.getbbox(text,stroke_width=stroke)
                    if hi_x<=lo_x or hi_y<=lo_y:
                        continue
                    img=Image.new("RGBA",(hi_x-lo_x,hi_y-lo_y),(0,0,0,0))
                    ImageDraw.Draw(img).text((-lo_x,-lo_y),text,font=font,
                        fill=(bgr[2],bgr[1],bgr[0],255),stroke_width=stroke,stroke_fill=(0,0,0,255))
                    rgba=np.asarray(img)
                    tile=(rgba[...,:3][...,::-1].copy(),rgba[...,3:4].astype(np.uint16),lo_x,lo_y)
                    self._text_tiles[key]=tile
                    if len(self._text_tiles)>256:
                        self._text_tiles.popitem(last=False)
                else:
                    self._text_tiles.move_to_end(key)
                color,alpha,lo_x,lo_y=tile
                x,y=int(x)+lo_x,int(y)+lo_y
                x1,y1=max(0,x),max(0,y)
                x2,y2=min(frame.shape[1],x+color.shape[1]),min(frame.shape[0],y+color.shape[0])
                if x2<=x1 or y2<=y1:
                    continue
                a=alpha[y1-y:y2-y,x1-x:x2-x]
                c=color[y1-y:y2-y,x1-x:x2-x].astype(np.uint16)
                dest=frame[y1:y2,x1:x2]
                dest[:]=((c*a+dest.astype(np.uint16)*(255-a)+127)//255).astype(np.uint8)
        else:
            for text, x, y, px, bgr in ops:
                cv2.putText(frame, text, (int(x), int(y + px * 0.75)),
                            cv2.FONT_HERSHEY_SIMPLEX, px / 26.0, bgr,
                            max(1, int(px) // 18), cv2.LINE_AA)


# ----------------------------------------------------------------------------
# 已学习数据管理（对应 PEFSL 的 DataFewShot，按字符串标签组织并支持存取）
# ----------------------------------------------------------------------------
class LearnedData:
    def __init__(self):
        self.reset()

    def reset(self):
        self.labels = []      # list[str] 类别标签
        self.shots = []       # list[np.ndarray(n_i,D)] 各类别特征样本
        self.class_fracs = [] # list[list[float]] 每类历次注册框的相对尺寸(占画面短边比例)
        self.class_ars = []   # list[list[float]] 每类历次注册框的宽高比(w/h):
                              #   识别窗口按它生成矩形——宽扁/竖长物体用正方形
                              #   窗口会含大片背景, 相似度被稀释、窗口对不准
        self.thumbs = []      # list[np.ndarray(64,64,3)] 各类别首次注册时的缩略图
        self.visual_thumbs = [] # 每类最多8个学习外观，供不同录入角度的独立核验
        self.reference_views = []  # Complete, aspect-preserving registration ROIs.
        self.generic_classes = []  # Old V2 archives retain their acquisition policy.
        self.class_hists = [] # list[np.ndarray(512,)] 各类别 HSV 颜色直方图(L1归一)
        self.class_bgs = []   # list[np.ndarray(D,)] 框外背景的原始平均特征，比较时再统一减均值/归一化
        self.class_shapes = [] # list[np.ndarray(2,)] 各类别注册窗口的形状描述子
                               #   (边缘密度, 灰度对比度): 识别窗口的轮廓/纹理量
                               #   必须与注册时相当——弱化颜色、强化几何/轮廓判据
        self.neg_feats = None  # 用户标记的负样本特征(误检的背景区域), (N,D) 或 None
        self._legacy_mean = None  # Fixed compatibility calibration after generic enrolment.
        self._invalidate()

    def _invalidate(self,reference_views_changed=True):
        self._mean_feat = None
        self._class_means = None
        self._normalized_shots = None
        self._uncentered_shots = None
        if not reference_views_changed:
            return  # Negative features cannot change learned image references.
        self._visual_refs = {}
        self._body_shapes = {}
        self._pattern_refs = {}
        self._colour_geometry_refs = {}
        self._scene_refs = {}
        self._dense_refs = {}
        self._spatial_refs = {}
        self._generic_pattern_groups = {}

    def class_visual_refs(self, ci):
        if ci not in self._visual_refs:
            references=[]
            for thumb in self.class_reference_views(ci):
                for flipped in (False,True):
                    for rotation in range(4):
                        image=np.ascontiguousarray(np.rot90(thumb[:,::-1] if flipped else thumb,rotation))
                        descriptor=describe_appearance(image,(0,0,image.shape[1],image.shape[0]))
                        if descriptor is not None:
                            references.append(descriptor)
            self._visual_refs[ci]=references
        return self._visual_refs[ci]

    def class_body_shapes(self, ci):
        if ci not in self._body_shapes:
            contours = [bright_body_contour(image)[0] for image in self.visual_thumbs[ci]]
            self._body_shapes[ci] = [c for c in contours if c is not None]
        return self._body_shapes[ci]

    def class_pattern_refs(self, ci):
        if ci not in self._pattern_refs:
            self._pattern_refs[ci] = registration_patterns(self.class_reference_views(ci))
        return self._pattern_refs[ci]

    def is_generic(self, ci):
        return ci < len(self.generic_classes) and bool(self.generic_classes[ci])

    def class_reference_views(self, ci):
        if ci < len(self.reference_views) and self.reference_views[ci]:
            return self.reference_views[ci]
        return self.visual_thumbs[ci]

    def class_scene_refs(self,ci,method='orb'):
        key=(ci,method)
        if key not in self._scene_refs:
            self._scene_refs[key]=scene_reference_features(self.class_reference_views(ci),method)
        return self._scene_refs[key]

    def class_dense_refs(self,ci):
        if ci not in self._dense_refs:
            self._dense_refs[ci]=dense_reference_features(self.class_reference_views(ci))
        return self._dense_refs[ci]

    def class_spatial_refs(self,ci,patch_size=24):
        key=ci if patch_size==24 else (ci,patch_size)
        if key not in self._spatial_refs:
            self._spatial_refs[key]=spatial_reference_features(self.class_reference_views(ci),patch_size=patch_size)
        return self._spatial_refs[key]

    def class_colour_geometry_refs(self,ci):
        if ci not in self._colour_geometry_refs:
            self._colour_geometry_refs[ci]=colour_registration_features(
                self.class_reference_views(ci), preserve_aspect=self.is_generic(ci))
        return self._colour_geometry_refs[ci]

    @staticmethod
    def _prepare_visual_thumb(thumb):
        if thumb is None:
            return None
        image = np.asarray(thumb)
        if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
            return None
        if not np.any(image):
            return None  # 旧存档中的全零占位图不代表学习过的外观
        image = np.clip(image, 0, 255).astype(np.uint8)
        if image.shape[:2] != (64, 64):
            image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(image).copy()

    def _add_visual_thumb(self, ci, thumb):
        image = self._prepare_visual_thumb(thumb)
        if image is None:
            return
        bank = self.visual_thumbs[ci]
        if any(np.array_equal(image, saved) for saved in bank):
            return
        # 保留首次外观以及最近7次录入，避免重复录入挤掉最初的一面。
        if len(bank) >= 8:
            del bank[1]
        bank.append(image)

    def normalized_shots(self):
        if self._normalized_shots is None:
            self._normalized_shots = feature_preprocess(
                np.concatenate(self.shots, axis=0), self.mean_feature())
        return self._normalized_shots

    def uncentered_shots(self):
        if self._uncentered_shots is None:
            shots=np.concatenate(self.shots,axis=0)
            self._uncentered_shots=feature_preprocess(shots,np.zeros(shots.shape[1],np.float32))
        return self._uncentered_shots

    def has_data(self):
        return len(self.labels) > 0

    def add_shots(self, label, feats, box_frac, thumb=None, hist=None, bg_feats=None,
                  box_ar=1.0, shape=None, reference_views=None, generic=False,
                  background_trusted=True):
        feats = np.asarray(feats, dtype=np.float32)
        if feats.ndim == 1:
            feats = feats[None]
        if generic and self.has_data() and self._legacy_mean is None \
                and any(not self.is_generic(ci) for ci in range(len(self.labels))):
            # Adding/upgrading a generic class must not recenter the existing
            # compatibility classifier or its background/negative models.
            self._legacy_mean = self.mean_feature().copy()
        # 保留背景特征原始尺度，与 shots 使用同一空间；_bg_gate 统一
        # 执行减均值后归一化，不能先单位化背景再减原始特征均值。
        bgv = None
        if background_trusted and bg_feats is not None and len(bg_feats) > 0:
            m = np.asarray(bg_feats, dtype=np.float32).mean(axis=0)
            n = float(np.linalg.norm(m))
            if n > 0:
                bgv = m
        if label in self.labels:
            i = self.labels.index(label)
            if generic:
                self.generic_classes[i] = True
                # Old automatic context may contain parts of this same object.
                self.class_bgs[i] = None
            self.shots[i] = np.concatenate((self.shots[i], feats), axis=0)
            self.class_fracs[i].append(float(box_frac))
            self.class_ars[i].append(float(box_ar))
            if shape is not None:      # 形状描述子取均值(多次注册融合)
                old = self.class_shapes[i] if i < len(self.class_shapes) else None
                sv = np.asarray(shape, dtype=np.float32)
                self.class_shapes[i] = sv if old is None else 0.5 * (old + sv)
            if self._prepare_visual_thumb(self.thumbs[i]) is None:
                image = self._prepare_visual_thumb(thumb)
                if image is not None:
                    self.thumbs[i] = image
            self._add_visual_thumb(i, thumb)
            if hist is not None:
                h = np.asarray(hist, dtype=np.float32).flatten()
                if self.class_hists[i] is None or float(self.class_hists[i].sum()) <= 0:
                    self.class_hists[i] = h
                else:   # 多次注册取均值（保持 L1 归一）
                    self.class_hists[i] = 0.5 * (self.class_hists[i] + h)
            if bgv is not None:     # 多次注册仍在原始特征空间融合
                old = self.class_bgs[i] if i < len(self.class_bgs) else None
                if old is None:
                    self.class_bgs[i] = bgv
                else:
                    self.class_bgs[i] = 0.5 * (old + bgv)
        else:
            self.labels.append(label)
            self.generic_classes.append(bool(generic))
            self.reference_views.append([])
            self.shots.append(feats)
            self.class_fracs.append([float(box_frac)])
            self.class_ars.append([float(box_ar)])
            self.class_shapes.append(np.asarray(shape, dtype=np.float32)
                                     if shape is not None else None)
            image = self._prepare_visual_thumb(thumb)
            self.thumbs.append(image if image is not None
                               else np.zeros((64, 64, 3), np.uint8))
            self.visual_thumbs.append([])
            self._add_visual_thumb(len(self.labels)-1, thumb)
            self.class_hists.append(np.asarray(hist, dtype=np.float32).flatten()
                                    if hist is not None else None)
            self.class_bgs.append(bgv)
        ci = self.labels.index(label)
        for image in (() if reference_views is None else reference_views):
            add_reference_view(self.reference_views[ci], image)
        self._invalidate()

    def class_hist(self, i):
        """第 i 类的 HSV 直方图；无数据返回 None（门控自动跳过）"""
        if 0 <= i < len(self.class_hists):
            h = self.class_hists[i]
            if h is not None and float(np.sum(h)) > 0:
                return h.astype(np.float32)
        return None

    def class_bg(self, i):
        """第 i 类框外背景的原始平均特征；无数据返回 None（门控自动跳过）。"""
        if 0 <= i < len(self.class_bgs):
            v = self.class_bgs[i]
            if v is not None and float(np.linalg.norm(v)) > 0:
                return np.asarray(v, dtype=np.float32)
        return None

    def total_shots(self, label):
        if label in self.labels:
            return len(self.shots[self.labels.index(label)])
        return 0

    def mean_feature(self):
        """基准均值特征（PEFSL 初始化阶段学习的 mean_repr 的对应物）"""
        if not self.has_data():
            return None
        if self._legacy_mean is not None:
            return self._legacy_mean
        if self._mean_feat is None:
            self._mean_feat = np.concatenate(self.shots, axis=0).mean(axis=0)
        return self._mean_feat

    def class_means(self):
        if self._class_means is None:
            self._class_means = np.stack([s.mean(axis=0) for s in self.shots], axis=0)
        return self._class_means

    def class_scan_fracs(self, default=0.4):
        """每个类别各自的推荐扫描窗口尺寸（该类注册框尺寸中位数, 占画面短边比例）。
        封顶 0.72：超过 72% 短边的整屏大窗是'背景冒充物体'的载体，不参与扫描。"""
        out = []
        for ci,row in enumerate(self.class_fracs):
            limits=(.03,2.) if self.is_generic(ci) else (.12,.72)
            out.append(float(np.clip(np.median(row), *limits)) if row else default)
        return out

    def class_ar(self, i, default=1.0):
        """第 i 类注册框的宽高比中位数(w/h)。识别窗口按它生成矩形——
        宽扁/竖长物体用正方形窗口会含大片背景, 特征被稀释、窗口对不准。"""
        if i < len(self.class_ars) and self.class_ars[i]:
            limits=(.02,50.) if self.is_generic(i) else (.25,4.)
            return float(np.clip(np.median(self.class_ars[i]), *limits))
        return default

    def class_shape(self, i):
        """第 i 类注册窗口的形状描述子 (边缘密度, 灰度对比度)；无则 None"""
        if i < len(self.class_shapes):
            return self.class_shapes[i]
        return None

    def add_negatives(self, feats):
        """把误检区域的特征登记为负样本(可多次累加)。识别时窗口若
        '比负样本还不像物体'就拒绝——用户标记一次即可排除这类误检。"""
        feats = np.asarray(feats, dtype=np.float32)
        if feats.ndim == 1:
            feats = feats[None]
        self.neg_feats = feats if self.neg_feats is None else \
            np.concatenate((self.neg_feats, feats), axis=0)
        self._invalidate(reference_views_changed=False)

    def num_negatives(self):
        return 0 if self.neg_feats is None else int(len(self.neg_feats))

    def save(self, path):
        if not self.has_data():
            return False, "nothing"
        counts = np.array([len(s) for s in self.shots], dtype=np.int64)
        feats = np.concatenate(self.shots, axis=0)
        payload = dict(schema_version=np.array(2, dtype=np.int64),
                       labels=np.array(self.labels),
                       counts=counts,
                       features=feats)
        m = max((len(r) for r in self.class_fracs), default=0)
        pad = np.full((len(self.labels), max(m, 1)), -1.0)
        for i, r in enumerate(self.class_fracs):
            pad[i, :len(r)] = r
        payload["class_fracs"] = pad
        ma = max((len(r) for r in self.class_ars), default=0)
        pada = np.full((len(self.labels), max(ma, 1)), -1.0, dtype=np.float32)
        for i, r in enumerate(self.class_ars):
            pada[i, :len(r)] = r
        payload["class_ars"] = pada
        if len(self.class_shapes) == len(self.labels) and self.labels:
            payload["class_shapes"] = np.stack([
                s if s is not None else np.zeros(2, np.float32)
                for s in self.class_shapes]).astype(np.float32)
        if self.neg_feats is not None and len(self.neg_feats):
            payload["neg_feats"] = np.asarray(self.neg_feats, np.float32)
        if len(self.thumbs) == len(self.labels) and self.thumbs:
            payload["thumbs"] = np.stack([
                self._prepare_visual_thumb(t) if self._prepare_visual_thumb(t) is not None
                else np.zeros((64, 64, 3), np.uint8) for t in self.thumbs])
        visual_counts = np.asarray([len(bank) for bank in self.visual_thumbs], np.int64)
        visual_images = [image for bank in self.visual_thumbs for image in bank]
        payload["visual_thumb_counts"] = visual_counts
        payload["visual_thumbs"] = (np.stack(visual_images) if visual_images
                                     else np.empty((0, 64, 64, 3), np.uint8))
        payload["generic_classes"] = np.asarray(self.generic_classes, np.bool_)
        if self._legacy_mean is not None:
            payload['legacy_mean']=self._legacy_mean.astype(np.float32)
        payload.update(pack_reference_views(self.reference_views))
        if len(self.class_hists) == len(self.labels) and self.labels:
            payload["class_hists"] = np.stack([
                h if h is not None else np.zeros(512, np.float32)
                for h in self.class_hists]).astype(np.float32)
        if len(self.class_bgs) == len(self.labels) and self.labels:
            payload["class_bgs"] = np.stack([
                b if b is not None else np.zeros(feats.shape[1], np.float32)
                for b in self.class_bgs]).astype(np.float32)
        path=os.fspath(path)
        if not path.endswith('.npz'):path += '.npz'
        ok,error,actual=save_archive_snapshot(path,payload)
        if not ok:return False,error
        self.last_saved_path=actual
        return True, f"{len(self.labels)} classes / {len(feats)} shots -> {actual}"

    def load(self, path):
        if not os.path.exists(path):
            return False, "missing"
        # Parse into a temporary object. A malformed optional reference bank
        # must not discard the current (possibly unsaved) learning session.
        candidate=LearnedData()
        try:
            with np.load(path) as payload:
                ok,message=candidate._load_payload(payload)
        except (OSError,EOFError,OverflowError,ValueError,KeyError,IndexError,TypeError,zipfile.BadZipFile) as exc:
            return False,f'corrupt: {exc}'
        if ok:
            self.__dict__.clear()
            self.__dict__.update(candidate.__dict__)
        return ok,message

    def _load_payload(self,d):
        if "schema_version" not in d.files or int(d["schema_version"].item()) != 2:
            d.close()
            return False, "旧版学习数据需重新采集（正样本可能混有背景，背景特征尺度不兼容）"
        labels = [str(s) for s in d["labels"]]
        raw_counts = d["counts"]
        if raw_counts.ndim != 1 or not np.isfinite(raw_counts).all():
            return False, "corrupt counts"
        with np.errstate(over='ignore', invalid='ignore'):
            counts = raw_counts.astype(np.int64)
            feats = np.asarray(d["features"], np.float32)
        if (counts.ndim!=1 or feats.ndim!=2 or feats.shape[1]<1
                or not labels or np.any(counts<=0)
                or not np.array_equal(raw_counts, counts)
                or not np.isfinite(feats).all()
                or len(labels) != len(counts)
                or feats.shape[0] != sum(int(c) for c in counts)):
            d.close()
            return False, "corrupt"
        # Validate optional arrays before committing any of this archive. Their
        # feature width follows the archive, rather than a particular backbone.
        feature_dim = feats.shape[1]
        for key, tail in (("class_hists", (512,)), ("class_bgs", (feature_dim,)),
                          ("class_shapes", (2,)), ("thumbs", (64, 64, 3))):
            if key not in d.files:
                continue
            values = d[key]
            if values.ndim == 0:
                return False, f"corrupt {key}"
            if values.shape[0] != len(labels):
                continue  # Preserve the older missing-metadata fallback.
            with np.errstate(over='ignore', invalid='ignore'):
                numeric = np.asarray(values, np.float32)
            if values.shape[1:] != tail or not np.isfinite(numeric).all():
                return False, f"corrupt {key}"
            if key in ("class_hists", "thumbs") and np.any(numeric < 0):
                return False, f"corrupt {key}"
            if key == "thumbs" and np.any(numeric > 255):
                return False, f"corrupt {key}"
        if "neg_feats" in d.files:
            with np.errstate(over='ignore', invalid='ignore'):
                negatives = np.asarray(d["neg_feats"], np.float32)
            if (negatives.ndim != 2 or negatives.shape[1] != feature_dim
                    or not np.isfinite(negatives).all()):
                return False, "corrupt neg_feats"
        self.reset()
        off = 0
        for lab, c in zip(labels, counts):
            self.labels.append(lab)
            self.shots.append(feats[off:off + c].astype(np.float32))
            off += c
        if "class_fracs" in d.files and d["class_fracs"].shape[0] == len(self.labels):
            self.class_fracs = [[float(v) for v in row if v >= 0] for row in d["class_fracs"]]
        elif "box_fracs" in d.files:  # 兼容旧版保存的平铺尺寸
            shared = [float(f) for f in d["box_fracs"]]
            self.class_fracs = [list(shared) for _ in self.labels]
        else:
            self.class_fracs = [[] for _ in self.labels]
        if "class_ars" in d.files and d["class_ars"].shape[0] == len(self.labels):
            self.class_ars = [[float(v) for v in row if v >= 0] for row in d["class_ars"]]
        else:
            self.class_ars = [[] for _ in self.labels]   # 旧数据: 回退方形窗口
        if "class_shapes" in d.files and d["class_shapes"].shape[0] == len(self.labels):
            self.class_shapes = [np.asarray(s, np.float32)
                                 if float(np.abs(np.asarray(s)).sum()) > 0 else None
                                 for s in d["class_shapes"]]
        else:
            self.class_shapes = [None for _ in self.labels]
        self.neg_feats = (np.asarray(d["neg_feats"], np.float32)
                          if "neg_feats" in d.files else None)
        if "thumbs" in d.files and d["thumbs"].shape[0] == len(self.labels):
            self.thumbs = [np.asarray(t, np.uint8) for t in d["thumbs"]]
        else:
            self.thumbs = [np.zeros((64, 64, 3), np.uint8) for _ in self.labels]
        self.visual_thumbs = [[] for _ in self.labels]
        valid_visual_bank = False
        if "visual_thumbs" in d.files and "visual_thumb_counts" in d.files:
            visual_images = np.asarray(d["visual_thumbs"])
            visual_counts = np.asarray(d["visual_thumb_counts"], np.int64)
            valid_visual_bank = (visual_images.ndim == 4
                                 and visual_images.shape[1:] == (64, 64, 3)
                                 and visual_counts.shape == (len(self.labels),)
                                 and np.all(visual_counts >= 0)
                                 and int(visual_counts.sum()) == len(visual_images))
            if valid_visual_bank:
                offset = 0
                for ci, count in enumerate(visual_counts):
                    for image in visual_images[offset:offset+int(count)]:
                        self._add_visual_thumb(ci, image)
                    offset += int(count)
        if not valid_visual_bank:
            # 兼容原有V2：只有首次缩略图，仍作为该类的一个有效外观。
            for ci, thumb in enumerate(self.thumbs):
                self._add_visual_thumb(ci, thumb)
        if "class_hists" in d.files and d["class_hists"].shape[0] == len(self.labels):
            self.class_hists = [h if float(np.sum(h)) > 0 else None
                                for h in d["class_hists"].astype(np.float32)]
        else:
            self.class_hists = [None] * len(self.labels)
        if "class_bgs" in d.files and d["class_bgs"].shape[0] == len(self.labels):
            self.class_bgs = [b if float(np.linalg.norm(b)) > 0 else None
                              for b in d["class_bgs"].astype(np.float32)]
        else:
            self.class_bgs = [None] * len(self.labels)
        self.generic_classes = ([bool(v) for v in d['generic_classes']]
                                if 'generic_classes' in d.files
                                and d['generic_classes'].shape == (len(labels),)
                                else [False] * len(labels))
        if 'legacy_mean' in d.files:
            centre=np.asarray(d['legacy_mean'],np.float32)
            if centre.shape!=(feats.shape[1],) or not np.isfinite(centre).all():
                d.close()
                return False,'invalid compatibility calibration'
            self._legacy_mean=centre.copy()
        elif any(self.generic_classes) and not all(self.generic_classes):
            self._legacy_mean=np.concatenate([shots for ci,shots in enumerate(self.shots)
                if not self.is_generic(ci)],axis=0).mean(axis=0)
        try:
            self.reference_views = unpack_reference_views(d, len(labels))
        except ValueError as exc:
            d.close()
            self.reset()
            return False, str(exc)
        self._invalidate()
        d.close()
        return True, f"{len(labels)} classes / {len(feats)} shots"


# ----------------------------------------------------------------------------
# PEFSL 分类器（NCM / KNN，与 FewShotModel.predict_class_* 同一套核心函数）
# ----------------------------------------------------------------------------
class PefslClassifier:
    def __init__(self, ctype="ncm", k=5):
        self.ctype = ctype
        self.k = k

    def predict_batch(self, feats, data):
        """原始特征 (N,D) -> 概率 (N,C)。复用 few_shot_model 的
        feature_preprocess / ncm / knn，逻辑与原始演示一致，
        但支持各类别样本数不等情况（knn 原函数允许）。"""
        mean = data.mean_feature()
        if data.labels and all(data.is_generic(ci) for ci in range(len(data.labels))):
            mean=np.zeros_like(mean)
        feats = np.atleast_2d(np.asarray(feats, dtype=np.float32))
        feats = feature_preprocess(feats, mean)
        n_cls = len(data.labels)
        if self.ctype == "ncm":
            shots = feature_preprocess(data.class_means(), mean)  # (C,D)
            probas = ncm(shots, feats)                             # (N,C)
        else:
            pts = np.concatenate(data.shots, axis=0)
            pts = feature_preprocess(pts, mean)
            targets = np.concatenate(
                [np.full(len(s), i, dtype=np.int64) for i, s in enumerate(data.shots)])
            n_pts = len(targets)
            kk = min(self.k, n_pts)
            if kk >= n_pts:  # 样本极少时退化为类频率
                cnt = np.bincount(targets, minlength=n_cls)
                probas = np.tile(cnt / max(1, cnt.sum()), (len(feats), 1))
            else:
                probas = knn(pts, feats, targets, kk)              # (N,C)
        probas = np.asarray(probas, dtype=np.float64).reshape(len(feats), n_cls)
        return probas

    def cosine_batch(self, feats, data, topk=5):
        """原始特征 (N,D) -> 与各类注册样本的相似度得分 (N,C)。
        得分 = 0.7×最高相似样本 + 0.3×Top-K 均值：
        - 纯 Top-K 均值会稀释真物体——注册时每帧含多个尺度/角度的增强
          样本(原图/中心裁剪/放大/旋转)，当前窗口只与'同尺度的那几个'
          高度匹配(峰值 0.7+)，其余不匹配样本会把均值拉到 0.5 以下，
          真物体就在眼前也过不了门槛；
        - 纯最大值(1-NN)又太松，单个背景 patch 偶然高相似就会通过。
        两者加权兼顾召回与抗撞脸。"""
        mean = data.mean_feature()
        feats = np.atleast_2d(np.asarray(feats, dtype=np.float32))
        original=feats
        generic=[ci for ci in range(len(data.labels)) if data.is_generic(ci)]
        raw_sims=None
        if generic:
            # A mean fitted only on positive examples is not a background/base
            # dataset mean: subtracting it removes the sole class's identity.
            raw_sims=feature_preprocess(original,np.zeros(original.shape[1],np.float32)) @ data.uncentered_shots().T
        if generic and len(generic)==len(data.labels):
            # Every class uses raw_sims below. Do not prepare and multiply an
            # unused centered bank on every foreground/background query.
            sims=raw_sims
        else:
            feats = feature_preprocess(feats, mean)      # (N,D) 单位球归一化
            pts = data.normalized_shots()               # (S,D), invalidated on learning/load
            sims = feats @ pts.T                        # (N,S)
        out = np.zeros((len(feats), len(data.labels)), dtype=np.float64)
        off = 0
        for i, s in enumerate(data.shots):
            k = min(topk, len(s))
            part = sims[:, off:off + len(s)]
            if i in generic:
                part=raw_sims[:,off:off+len(s)]
            if k < len(s):
                part = np.partition(part, -k, axis=1)[:, -k:]
            out[:, i] = 0.7 * part.max(axis=1) + 0.3 * part.mean(axis=1)
            off += len(s)
        return out

    def smooth(self, prev, cur):
        """PEFSL 演示同款指数滑动平均系数：ncm 0.85/0.15，knn 0.95/0.05"""
        alpha = 0.85 if self.ctype == "ncm" else 0.95
        if prev is None or prev.shape != cur.shape:
            return cur.copy()
        return alpha * prev + (1.0 - alpha) * cur


# ----------------------------------------------------------------------------
# 摄像头与骨干网络
# ----------------------------------------------------------------------------
def init_camera(args):
    """打开摄像头：依次尝试 MSMF / DirectShow / 默认后端，每次打开后
    试读一帧确认真能出图（MSMF 在设备被占用/驱动卡死时能 open 但
    grabFrame 一直报 -1072875772，此时换 DirectShow 后端通常可救活）。"""
    tried = []
    for pref, name in ((cv2.CAP_MSMF, "MSMF"), (cv2.CAP_DSHOW, "DSHOW"),
                       (cv2.CAP_ANY, "ANY")):
        if name in tried:
            continue
        tried.append(name)
        try:
            cap = cv2.VideoCapture(args.camera_id, pref)
        except Exception:
            cap = None
        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            print(f"[camera] backend={name} 打开失败")
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_resolution[1])
        ok, _ = cap.read()            # 试读一帧：open 成功 ≠ 能出图
        if ok:
            actual = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            print(f"[camera] id={args.camera_id} backend={name} "
                  f"请求={args.camera_resolution} 实际={actual} "
                  f"reported_fps={cap.get(cv2.CAP_PROP_FPS):.1f} exposure={cap.get(cv2.CAP_PROP_EXPOSURE):.1f}")
            return cap
        cap.release()
        print(f"[camera] backend={name} 打开成功但读不出帧(被占用/驱动卡死), 换后端重试")
    raise RuntimeError(
        f"无法从摄像头 {args.camera_id} 读取画面（试试 --camera-id 1，"
        f"或重启摄像头设备/关闭占用摄像头的程序后重试）")


def load_backbone(args):
    """加载 PEFSL 骨干网络（pytorch 框架, ResNet-9, 输出 80 维特征）"""
    # Tiny crops lose time scheduling dozens of worker threads. Keep the UI
    # responsive; this is configurable for machines with a different workload.
    threads = max(1,int(getattr(args,"cpu_threads",2)))
    cv2.setNumThreads(threads)
    if args.device_pytorch == "cpu":
        import torch
        torch.set_num_threads(threads)
    print(f"[runtime] CPU/OpenCV threads={threads}")
    specs = {"type": "pytorch", "device": args.device_pytorch,
             "model_name": args.backbone, "use_strides": True,
             "weight": args.path_pytorch_weight}
    backbone = get_model(specs)
    print(f"[backbone] {args.backbone} @ {args.device_pytorch} "
          f"weights={args.path_pytorch_weight}")
    return backbone


# ----------------------------------------------------------------------------
# 主应用：A 模式（框选学习）/ B 模式（框选识别 + 自动扫描）
# ----------------------------------------------------------------------------
class App:
    def __init__(self, args, backbone, data, clf, renderer):
        self.args = args
        self.backbone = backbone
        self.data = data
        self.clf = clf
        self.R = renderer
        self.W, self.H = args.output_resolution
        self.in_size = args.resolution_input
        self.bs = self.H / 600.0               # 界面元素基准缩放
        self.frame = np.zeros((self.H, self.W, 3), np.uint8)
        self.text_ops = []

        self.mode = ""                         # 由 run() 中 set_mode 初始化
        self.interaction = "idle"             # idle / label / collect
        self.drawing = False
        self.box_start = None
        self.box_cur = None
        self.pending_box = None
        self.pending_label = ""
        self.collect_feats = []
        self.collect_hists = []
        self.collect_bg_feats = []
        self.collect_bg_hists = []   # 框外环带的 HSV 直方图(判断'框里是不是背景')
        self.collect_n = 0
        self.collect_capture = None

        self.manual_box = None                # B 模式手选识别框
        self.manual_cosv = None               # 手选框识别的余弦相似度向量
        self.auto_scan = True                 # B 模式默认开启自动定位（框出识别到的物体）
        self.scan_cosv = None                 # 自动定位的 EMA 余弦相似度向量(逐类最佳)
        self.scan_targets = []                # 多目标列表: {box,label,cls,cos,asize,ema,fail,pend}
        # 自动负样本: 同一块地方被反复释放的计数(见 _auto_negative)
        self._fp_hits = {}
        self.scan_result = None               # 派生显示: 最佳目标 (box, label, cos)
        self.bg_model = None                  # 基准背景模型(空场景多帧平均, float32 HxWx3)
        self.raw_frame = None                 # 最近一帧**未叠加**的相机原图(按 d 存证用)
        self.bg_frames = []                   # 背景采样中的累积帧
        self.bg_target = 0                    # 本次背景采样目标帧数(0 表示未在采样)
        self._diag_pending = False            # 按 d 后置位, 帧末落盘诊断证据
        self._diag_writer = None
        self._fgv = None                      # 上次扫描各窗口的前景占比(分级门槛用)
        self._scan_log_t = 0.0
        self.frame_cosv = None                # 整幅画面识别的 EMA 余弦相似度向量
        self._diag_localized = {}
        self._boundary_search_rounds = {}
        self._async_search = None
        self._search_epoch = 0
        self._search_data_snapshot = None
        self._search_app = None
        self._deferred_search_result = None
        self._generic_identity_hints = {}
        self._generic_identity_hint_bank = {}
        self._generic_template_preferences = {}
        self._identity_task_states = {}
        self._next_search_time = 0.
        self._search_times = deque()
        self.search_fps = 0.
        self._search_last_ms = self._search_age_ms = 0.
        self._search_job_count = 0
        self.rng = np.random.default_rng()

        self.frame_idx = 0
        self.fps = 0.0
        self._frame_times = deque()
        self._frame_time_total = 0.0
        self._perf_current = {}
        self.status = ""
        self.quit = False

    # ---------------- 小工具 ----------------
    def t(self, key):
        return self.R.t(key)

    def msg(self, zh, en):
        return zh if self.R.lang == "zh" else en

    def text(self, s, x, y, px, color=WHITE):
        self.text_ops.append((s, x, y, px, color))

    def set_status(self, s):
        self.status = s

    def _data_changed(self):
        self._invalidate_search()
        self.manual_cosv = None
        self.scan_cosv = None
        self.scan_result = None
        self.scan_targets = []
        self.frame_cosv = None
        self._diag_localized = {}
        self._diag_reject = {}
        self._diag_reject_history = {}
        self._diag_reject_frame = self.frame_idx
        self._diag_stage = None
        self._diag_generic_candidates = {}
        self._diag_generic_source_work = {}
        self._diag_generic_source_time = None
        self._diag_generic_source_frame = None
        self._boundary_search_rounds = {}

    def _invalidate_search(self):
        self._search_epoch = getattr(self,"_search_epoch",0)+1
        self._search_last_ms = self._search_age_ms = 0.
        self._search_last_kind = None
        self._search_error_streak = 0
        self._search_last_error = None
        self._search_times.clear()
        self.search_fps = 0.
        self._search_data_snapshot = None
        self._deferred_search_result = None
        self._generic_identity_hints = {}
        self._generic_identity_hint_bank = {}
        self._generic_template_preferences = {}
        self._identity_task_states = {}
        self._generic_colour_reference_cache = {}
        self._next_search_time = 0.

    def class_color(self, idx):
        return CLASS_COLORS[idx % len(CLASS_COLORS)]

    def resolve_label(self, name):
        """空标签自动命名 obj1/obj2/...，并保证唯一"""
        name = name.strip()
        if not name:
            i = len(self.data.labels) + 1
            name = f"obj{i}"
            while name in self.data.labels:
                i += 1
                name = f"obj{i}"
        return name

    # ---------------- 输入 ----------------
    def handle_key(self, k):
        if k == 255:  # waitKey 无按键
            return
        if self.interaction == "label":
            self._key_label(k)
            return
        if self.interaction == "collect":
            if k == ord("q") or k == 27:  # 中断采集
                self.interaction = "idle"
                self.pending_box = None
                self.collect_capture = None
                self._collect_aug_frame = None
                self._collect_aug_box = None
                self.collect_feats, self.collect_hists, self.collect_n, \
                    self.collect_bg_feats, self.collect_bg_hists = [], [], 0, [], []
                self.set_status(self.t("hint_a"))
            elif k in (ord("+"), ord("="), ord("-")):  # 采集中也可调整
                self._adjust_shots(k)
            return
        if k == ord("q") or k == 27:
            self.quit = True
            return
        c = chr(k).lower() if 32 <= k <= 126 else ""
        if c == "a":
            self.set_mode("A")
        elif c == "b":
            self.set_mode("B")
        elif c == "s":
            self._invalidate_search()
            self.auto_scan = not self.auto_scan
            self.manual_box = None
            self.set_status(self.t("scan_on") if self.auto_scan else self.t("scan_off"))
        elif c == "c":
            self._invalidate_search()
            self.manual_box = None
            self.manual_cosv = None
            self.scan_result = None
            self.scan_targets = []
            self.set_status(self.t("box_cleared"))
        elif c == "r":
            self.data.reset()
            self._data_changed()
            self.manual_box = None
            self._fp_hits = {}            # 自动负样本的计数一并清空
            self.set_status(self.t("reset"))
            print("[data] reset")
        elif c == "w":
            ok, info = self.data.save(self.args.save_path)
            if ok:
                self.args.save_path=getattr(self.data,'last_saved_path',self.args.save_path)
                self._save_bg_model()
            s = f"{self.t('saved')}: {info}" if ok else (
                self.t("nothing_save") if info=='nothing' else self.msg(f'保存失败：{info}',f'Save failed: {info}'))
            self.set_status(s)
            print("[data] save:", info)
        elif c == "l":
            ok, info = self.data.load(self.args.save_path)
            if ok:
                self._load_bg_model()
                self._data_changed()
            s = f"{self.t('loaded')}: {info}" if ok else (
                self.t("no_save") if info == "missing" else info)
            self.set_status(s)
            print("[data] load:", info)
        elif c == "k":
            self._toggle_bg_sample()
        elif c == "n":
            self._mark_negative()
        elif c == "d":
            # 置标志、留到本帧"画完之后"再落盘——这样存下来的 frame 图才带
            # 框和屏上诊断文字(handle_key 此刻 frame 还是刚读入的相机原图)
            self._diag_pending = True
        elif k in (ord("+"), ord("="), ord("-")):
            self._adjust_shots(k)

    def _adjust_shots(self, k):
        """运行中用 +/- 调整每次框选采集的帧数（步长 5，范围 5~200）"""
        step = -5 if k == ord("-") else 5
        self.args.shots_per_box = int(np.clip(self.args.shots_per_box + step, 5, 200))
        self.set_status(self.t("shots_set") % self.args.shots_per_box)
        print(f"[config] shots_per_box -> {self.args.shots_per_box}")

    def _mark_negative(self):
        """把框选区域登记为负样本(误检背景)，立即从识别中排除。
        用法：B 模式下鼠标框出误检区域，再按 n。可多次标记累加。"""
        if self.raw_frame is None:
            self.set_status(self.msg('当前没有有效摄像头画面，未添加负样本',
                                     'No current camera frame; no negative sample added.'))
            return
        b = self.manual_box
        if b is None and self.scan_targets:
            b = max(self.scan_targets, key=lambda t: t["cos"])["box"]
        if b is None:
            self.set_status(self.msg(
                "先用鼠标框出误检的背景区域, 再按 n 标记为负样本",
                "Drag over the false-positive area, then press n"))
            return
        crop = crop_resize(self.frame, b, self.in_size)
        feats = self.backbone(preprocess_batch(crop[None]))
        self.data.add_negatives(feats)
        n = self.data.num_negatives()
        # 联动背景模型：把误检区域直接吸收进基准背景, 该区域从此不再是
        # '前景', 双重抑制同一处的重复误检(特征黑名单 + 背景排除模板)
        if self.bg_model is not None:
            x1, y1 = max(0, int(b[0])), max(0, int(b[1]))
            x2, y2 = min(self.W, int(b[2])), min(self.H, int(b[3]))
            reg = self.frame[y1:y2, x1:x2]
            if reg.size:
                self.bg_model[y1:y2, x1:x2] = reg.astype(np.float32)
        # 立即撤掉压在该区域上的误检锁定——不然按 n 后还要等它几帧后自然失效,
        # 用户会以为'标了没用'。负样本门控生效后它本来也会掉, 这里只是即时化。
        removed = [t for t in self.scan_targets if _same_obj(t["box"], b, 0.5)]
        if removed:
            self.scan_targets = [t for t in self.scan_targets if t not in removed]
            self._update_scan_display()
        self._data_changed()
        self.set_status(self.msg(
            f"已登记负样本第 {n} 个区域——该背景不再被识别(按 w 可持久化)",
            f"Negative sample #{n} added (press w to save)"))
        print(f"[neg] added #{n}")

    def _auto_negative(self, box):
        """自动负样本（默认关闭）: 显式开启后，同一块地方被反复释放，登记其特征
        登记成负样本——与用户按 n 等效, 按 w 持久化。
        为什么需要: 背景白衣服/白桌面会**反复**"锁上几帧→被释放", 而用户
        不可能守着屏幕一次次按 n。触发刻意保守: 只对白/灰类计, 同一块地方
        (中心按画面 1/24 分桶)累计 3 次, 总数上限 12 个, 按 r 清空。
        走 _neg_gate 的相对+绝对规则, 若真物体被误伤, 把 --auto-neg-hits 设 0
        即可整体关闭。"""
        need = int(getattr(self.args, "auto_neg_hits", 0) or 0)
        if need <= 0:
            return
        key = ((int(box[0]) + int(box[2])) // 2 // 24,
               (int(box[1]) + int(box[3])) // 2 // 24)
        self._fp_hits[key] = self._fp_hits.get(key, 0) + 1
        if self._fp_hits[key] < need:
            return
        self._fp_hits[key] = 0
        if self.data.neg_feats is not None and len(self.data.neg_feats) >= 12:
            return
        crop = crop_resize(self.frame, box, self.in_size)
        self.data.add_negatives(self.backbone(preprocess_batch(crop[None])))
        self._invalidate_search()
        # 注意: 这里**绝不能**调 _data_changed()——它会把 scan_targets 整个清空,
        # 而本函数正是在遍历 scan_targets 的过程中被调用的, 调用方返回后还要
        # `scan_targets.remove(tgt)`; 列表被清空 -> ValueError 直接闪退(用户实测:
        # 第 3 次自动负样本时崩溃, 因为 --auto-neg-hits 默认就是 3)。
        # 只需让相似度显示缓存失效, 目标列表保持不动, 下一帧用新负样本重评即可。
        self.manual_cosv = None
        self.scan_cosv = None
        self.frame_cosv = None
        print(f"[neg] auto n={self.data.num_negatives()} "
              f"box={tuple(int(v) for v in box)}", flush=True)

    def _save_diag(self):
        """按 d: 一键把**这一帧的证据**落盘, 供离线排查。
        存三样(同一次按键、同一帧):
          1. <tag>_frame.png —— 完整渲染帧(带框 + 屏上诊断文字), 看"当时屏幕上是什么样"
          2. <tag>_raw.png   —— 未叠加的相机原图, 便于离线复算各道门控(红框/文字不污染)
          3. diag.txt        —— 逐类门控数值 + 卡在哪一关 + 目标/负样本状态(追加)
        比录屏精确(无压缩、无缩放), 也比录屏便宜得多。"""
        out = getattr(self.args, "diag_dir", "") or os.path.join(SCRIPT_DIR, "bbox_diag")
        tag = f"{time.strftime('%H%M%S')}_{int(self.frame_idx):06d}_{uuid.uuid4().hex[:8]}"
        images = [(name, img.copy()) for name, img in
                  (("frame", self.frame), ("raw", self.raw_frame)) if img is not None]
        images.extend(getattr(self,'_diag_extra_images',()))
        # ---- 数值快照 ----
        labels = self.data.labels
        whites = self._white_flags()
        L = []
        L.append(f"=== diag {time.strftime('%Y-%m-%d %H:%M:%S')} "
                 f"frame={self.frame_idx} fps={self.fps:.1f} mode={self.mode} "
                 f"auto_scan={self.auto_scan} ===")
        L.append(f"[学习与代码绑定] {diagnostic_context(self)}")
        L.append(f"[确认模板重找] enabled={getattr(self.args,'confirmed_template_reacquire',False)} "
                 f"choices={self._generic_template_preferences}")
        if getattr(self,'_diag_history_metadata',None):
            L.append(f"[连续原图索引] {self._diag_history_metadata}")
        for i, lab in enumerate(labels):
            fractions=self.data.class_fracs[i]
            fraction=f'{fractions[0]:.2f}' if fractions else 'unknown'
            shape=self.data.class_shape(i)
            shape=None if shape is None else np.round(np.asarray(shape),3).tolist()
            L.append(f"class[{i}] '{lab}' white={bool(whites[i])} generic={self._generic_class(i)} "
                     f"complete_refs={len(self.data.reference_views[i])} "
                     f"background_reference={self.data.class_bg(i) is not None} "
                     f"frac={fraction} "
                     f"ar={self.data.class_ar(i):.2f} "
                     f"shape={shape} "
                     f"shots={self.data.total_shots(lab)}")
        st = getattr(self, "_diag_stage", None)
        if st is not None and st.shape[1] == len(labels):
            L.append("[best 原始/色门/背门] " + "  ".join(
                f"{labels[i]}:{st[0, i] * 100:.0f}/{st[1, i] * 100:.0f}/"
                f"{st[2, i] * 100:.0f}" for i in range(len(labels))))
        rj = getattr(self, "_diag_reject", None) or {}
        if rj:
            L.append("[卡在哪] " + "  ".join(
                f"{(labels[int(k)] if 0 <= int(k) < len(labels) else k)}:{v}"
                for k, v in rj.items()))
        for ci, evidence in self._diag_localized.items():
            L.append(f"[贴合证据] {labels[ci]} box={evidence['box']} "
                     f"evidence_frame={evidence.get('frame', 'unknown')} "
                     f"age_frames={self.frame_idx-evidence['frame'] if 'frame' in evidence else 'unknown'} "
                     f"raw={evidence['raw']} color={evidence['color']} "
                     f"background={evidence['background']} negative={evidence['negative']} "
                     f"valid_views={evidence['valid_views']}/{len(evidence['raw'])} "
                     f"lock_views={evidence.get('lock_views', '-')} "
                     f"internal_structure={evidence.get('internal_structure')} "
                     f"score={evidence['score']} "
                     f"reason={evidence['reason']}")
            if evidence.get("boundary_view_refinement"):
                L.append(f"[边界内缩复核] {labels[ci]} {evidence['boundary_view_refinement']}")
        L.append(f"[耗时 ms] {self._perf_current}")
        if self.scan_targets:
            for t in self.scan_targets:
                L.append(f"target '{t['label']}' cls={t['cls']} cos={t['cos']:.2f} "
                         f"pend={t['pend']} verified={t.get('verified', False)} "
                         f"fail={t.get('fail', 0)} boundary={t.get('boundary_occupancy')} "
                         f"boundary_age={t.get('boundary_age', 0)} coasting={t.get('coasting', False)} "
                         f"registered_identity={t.get('registered_identity_similarity')} "
                         f"registered_pattern={t.get('registered_pattern_similarity')} "
                         f"identity_age_frames={self.frame_idx-t.get('identity_evidence_frame', self.frame_idx)} "
                         f"identity_blocked={t.get('identity_blocked', False)} "
                         f"boundary_source={t.get('boundary_source', 'endpoints')} "
                         f"boundary_pending={t.get('boundary_pending', False)} "
                         f"admission_pending={t.get('admission_pending', False)} "
                         f"colour_boundary={t.get('colour_boundary', False)} "
                         f"generic_geometry_admission={t.get('generic_geometry_admission',False)} "
                         f"registered_dense_alignment={t.get('registered_dense_alignment')} "
                         f"registered_spatial_alignment={t.get('registered_spatial_alignment')} "
                         f"generic_hidden_motion={t.get('generic_hidden_motion',False)} "
                         f"generic_motion_box={t.get('motion_box')} "
                         f"proposal_source={t.get('proposal_source')} "
                         f"colour_evidence={t.get('colour_boundary_evidence')} "
                         f"identity_blur_bridge={t.get('identity_blur_bridge', False)} "
                         f"hidden_motion_age_frames={self.frame_idx-t.get('hidden_motion_frame', -1000)} "
                         f"motion_quality={t.get('motion_quality', 0.):.3f} "
                         f"motion_coverage={t.get('motion_coverage')} "
                         f"endpoint_support={t.get('endpoint_support')} "
                         f"body_shape_distance={t.get('body_shape_distance')} "
                         f"body_elongation_ratio={t.get('body_elongation_ratio')} "
                         f"body_fraction={t.get('body_fraction')} "
                         f"box={tuple(int(v) for v in t['box'])}")
        else:
            L.append("targets=0")
        if getattr(self,'_diag_colour_candidates',None):
            age=(time.perf_counter()-getattr(self,'_diag_colour_source_time',time.perf_counter()))*1000
            L.append(f"[完整彩色候选] source_age_ms={age:.1f} {self._diag_colour_candidates}")
        if getattr(self,'_diag_generic_candidates',None):
            source_time=getattr(self,'_diag_generic_source_time',None)
            source_age=None if source_time is None else round((time.perf_counter()-source_time)*1000,1)
            L.append(f"[通用参考候选] source_frame={getattr(self,'_diag_generic_source_frame',None)} "
                     f"source_age_ms={source_age} {self._diag_generic_candidates}")
        if getattr(self,'_diag_generic_source_work',None):
            source_time=getattr(self,'_diag_generic_source_time',None)
            source_age=None if source_time is None else round((time.perf_counter()-source_time)*1000,1)
            L.append(f"[通用来源执行] source_frame={getattr(self,'_diag_generic_source_frame',None)} "
                     f"source_age_ms={source_age} {self._diag_generic_source_work}")
        for ci,state in getattr(self,'_identity_task_states',{}).items():
            if 0<=ci<len(labels):
                age=self.frame_idx-state.get('frame',self.frame_idx)
                L.append(f"[最近身份任务] {labels[ci]} age_frames={age} current_frame={age==0} {state}")
        for ci,previous in getattr(self,'_diag_reject_history',{}).items():
            if ci not in rj and 0<=ci<len(labels):
                L.append(f"[历史核验原因] {labels[ci]} age_frames={self.frame_idx-previous['frame']} "
                         f"reason={previous['reason']}")
        L.append(f"negatives={self.data.num_negatives()} bg_model="
                 f"{'yes' if self.bg_model is not None else 'no'} "
                 f"manual_box={self.manual_box}")
        text = "\n".join(L)
        if self._diag_writer is None:
            self._diag_writer = DiagnosticWriter()
        queued = self._diag_writer.submit(out, tag, images, text)
        self.set_status(self.msg("诊断正在后台保存" if queued else "诊断保存队列繁忙，请稍后再按 d",
                                 "Saving diagnostics" if queued else "Diagnostic queue full; try again"))

    def _bg_model_path(self):
        """背景模型存档路径(与学习数据同目录, 便于一起搬运)"""
        return self.args.save_path + ".bg.npy"

    def _save_bg_model(self):
        """把基准背景模型随学习数据一起持久化。原先背景模型只存在内存里,
        重启即丢——用户按 k 采过背景、重启后又得重采, 静态背景(书架隔板/
        墙上标签)的误检就会'复发'。存 float16 省一半体积。"""
        p = self._bg_model_path()
        if self.bg_model is None:
            try:
                os.remove(p)          # 数据里没有背景模型了, 存档也一起清掉
            except OSError:
                pass
            return
        try:
            np.save(p, self.bg_model.astype(np.float16))
        except OSError as e:
            print("[bg] save failed:", e)

    def _load_bg_model(self):
        """载入与学习数据配套的背景模型; 分辨率不符则忽略(换了输出尺寸)"""
        self.bg_model=None
        # An unfinished capture belongs to the previous load/session too.
        self.bg_target=0
        self.bg_frames=[]
        p = self._bg_model_path()
        if not os.path.exists(p):
            return
        try:
            m = np.load(p,allow_pickle=False)
            if not isinstance(m,np.ndarray):
                if hasattr(m,'close'):m.close()
                raise ValueError('background must be an array')
            m=m.astype(np.float32)
        except (OSError, ValueError, EOFError, OverflowError, zipfile.BadZipFile) as e:
            print("[bg] load failed:", e)
            return
        if m.shape == (self.H,self.W,3) and np.isfinite(m).all():
            self.bg_model = m
            print("[bg] model loaded (persisted)")
        else:
            print("[bg] model ignored (shape mismatch)")

    def _toggle_bg_sample(self):
        """k 键：开始/取消背景采样（采集 30 帧空场景生成基准背景模型）。
        注：原本建议绑 b，但 b 已是'切换识别模式'，故改用 k。"""
        self._invalidate_search()
        if self.bg_target > 0:
            self.bg_target = 0
            self.bg_frames = []
            self.set_status(self.msg("已取消背景采样", "Background sampling cancelled"))
            return
        self.bg_frames = []
        self.bg_target = 30
        self.set_status(self.msg(
            "背景采样中：请保持画面内没有目标物体(约1秒)...",
            "Sampling background: keep the scene empty (~1s)..."))

    def _tick_bg_sample(self):
        """背景模型维护：采样期累加取平均；建成后按 1% 步长缓慢自适应更新
        (抵消缓慢光照漂移)，且只在'当前帧几乎没有前景'时更新——防止把长
        时间停留的物体慢慢吸收进背景模型。"""
        f = self.frame.astype(np.float32)
        if self.bg_target > 0:
            self.bg_frames.append(f)
            if len(self.bg_frames) >= self.bg_target:
                self.bg_model = np.mean(np.stack(self.bg_frames, axis=0), axis=0)
                self.bg_frames = []
                self.bg_target = 0
                self._invalidate_search()
                self.set_status(self.msg(
                    "背景模型已建立(30帧平均)——静态背景不再进入识别",
                    "Background model built (30-frame average)"))
                print("[bg] model built")
            return
        if self.bg_model is None or self.frame_idx % 30 != 0:
            return           # 自适应更新降频到每 30 帧一次(前景掩码不必每帧算)
        m = self._fg_mask(f)
        if m is not None and float(m.mean()) / 255.0 < 0.02:
            # 吸收步长 0.01→0.005、频率 每10帧→每30帧: 原来约 0.5 秒吸 1%,
            # 物体在手里拿久一点就被慢慢吸进背景模型 -> 之后越来越认不出
            # ('用了 k 还没之前好使')。现在慢约 6 倍, 只用于抵消光照漂移。
            self.bg_model = 0.995 * self.bg_model + 0.005 * f

    def _fg_mask(self, frame_f):
        """前景掩码（在 1/4 分辨率下计算，像素运算量降为 1/16）：
        两帧各自直方图均衡(抗环境光缓慢变化) → 灰度差 > 25(滤掉呼吸/微动)
        → 3×3 开运算去噪 + 闭运算连通 → 剔除面积 < 画面 0.5% 的连通域
        → 掩码放大回原尺寸。"""
        if self.bg_model is None:
            return None
        s = 4
        w4, h4 = max(1, self.W // s), max(1, self.H // s)
        cur = cv2.resize(frame_f, (w4, h4), interpolation=cv2.INTER_AREA)
        bg = cv2.resize(self.bg_model, (w4, h4), interpolation=cv2.INTER_AREA)
        g1 = cv2.cvtColor(cur.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(bg.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(g1, g2)
        # 高亮区域(白窗/白墙/白桌面)阈值降到 12：白色物体(涂改带)与浅色背景
        # 的亮度差很小, 用 25 会把它当静态背景漏掉——白色物体漏检的主因
        th = np.where(g1 > 200, 12, 25).astype(np.uint8)
        m = (diff > th).astype(np.uint8) * 255
        # 边缘差分补充：轮廓处的梯度结构变化——即使亮度接近背景, 六边形
        # 轮廓边缘仍会变化, 白色涂改带由此被提为前景
        md = cv2.absdiff(
            cv2.magnitude(cv2.Sobel(g1, cv2.CV_32F, 1, 0, ksize=3),
                          cv2.Sobel(g1, cv2.CV_32F, 0, 1, ksize=3)),
            cv2.magnitude(cv2.Sobel(g2, cv2.CV_32F, 1, 0, ksize=3),
                          cv2.Sobel(g2, cv2.CV_32F, 0, 1, ksize=3)))
        m = cv2.bitwise_or(m, (md > 40).astype(np.uint8) * 255)
        k3 = np.ones((3, 3), np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k3)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k3)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        min_area = 0.005 * w4 * h4
        out = np.zeros_like(m)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                out[lab == i] = 255
        return cv2.resize(out, (self.W, self.H), interpolation=cv2.INTER_NEAREST)

    def _fg_frac(self, boxes, m=None):
        """每个候选窗口内的'前景像素占比'(0~1)；未建立背景模型时返回 None。
        m 可传入已算好的前景掩码, 避免重复计算。"""
        if self.bg_model is None:
            return None
        if m is None:
            m = self._fg_mask(self.frame.astype(np.float32))
        if m is None:
            return None
        out = np.zeros(len(boxes), np.float32)
        for k, b in enumerate(boxes):
            x1, y1 = max(0, int(b[0])), max(0, int(b[1]))
            x2, y2 = min(self.W, int(b[2])), min(self.H, int(b[3]))
            sub = m[y1:y2, x1:x2]
            if sub.size:
                out[k] = float(sub.mean()) / 255.0
        return out

    def _key_label(self, k):
        if k == 13:  # Enter 确认
            self.pending_label = self.resolve_label(self.pending_label)
            self.interaction = "collect"
            self.collect_feats, self.collect_hists, self.collect_n, \
                self.collect_bg_feats, self.collect_bg_hists = [], [], 0, [], []
            self.collect_capture = RegistrationCapture(self.pending_box)
            self.collect_reference_views = []
            self._collect_aug_frame = None
            self._collect_aug_box = None
            self._collect_last_capture_time = None
            self.set_status(self.t("collecting"))
        elif k == 27:  # ESC 取消
            self.interaction = "idle"
            self.pending_box = None
            self.pending_label = ""
            self.set_status(self.t("hint_a"))
        elif k in (8, 127):  # 退格
            self.pending_label = self.pending_label[:-1]
        elif 32 <= k <= 126 and len(self.pending_label) < 16:
            self.pending_label += chr(k)

    def on_mouse(self, event, x, y, flags, param):
        if self.interaction != "idle":
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.box_start = (x, y)
            self.box_cur = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.box_cur = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            b = norm_box(self.box_start, self.box_cur, self.W, self.H)
            side = min(b[2] - b[0], b[3] - b[1])
            if side < self.args.min_box_size:
                self.set_status(self.t("box_small"))
                return
            if self.mode == "A":
                self.pending_box = b
                self.pending_label = ""
                self.interaction = "label"
                self.set_status(self.t("hint_a"))
            else:  # B 模式：手选识别框
                self.manual_box = b
                self._invalidate_search()
                self.manual_cosv = None
                self.scan_result = None
                self.scan_targets = []
                if not self.data.has_data():
                    self.set_status(self.t("no_data_b"))

    def set_mode(self, m):
        if self.mode == m:
            return
        self.mode = m
        self._invalidate_search()
        self.manual_box = None
        self.manual_cosv = None
        self.scan_result = None
        self.scan_targets = []
        self.frame_cosv = None
        if m == "A":
            self.set_status(self.t("hint_a"))
        else:
            self.set_status(self.t("hint_b") if self.data.has_data() else self.t("no_data_b"))
        print(f"[mode] -> {m}")

    # ---------------- 每帧逻辑 ----------------
    def tick(self):
        self._generic_perf = {}
        self._tick_bg_sample()      # 背景模型: 采样/自适应更新
        # A 模式：采集注册框内多帧特征（PEFSL registration 状态的框选版）
        if self.interaction == "collect":
            if self.collect_n == 0 and getattr(self, 'collect_capture', None) is None:
                self.collect_capture = RegistrationCapture(self.pending_box)
                self.collect_reference_views = []
                self._collect_aug_frame = None
                self._collect_aug_box = None
                self._collect_last_capture_time = None
            captured = getattr(self,'_capture_time',None)
            if captured is not None and captured == getattr(self,'_collect_last_capture_time',None):
                return
            self._collect_last_capture_time = captured
            decision = self.collect_capture.observe(self.frame)
            if decision.action == 'pause':
                reason = ('持续未能找回物体，按Esc后重新框选' if self.collect_capture.lost
                          else '画面模糊，请保持清晰；恢复后自动继续' if decision.reason=='blur'
                          else '正在重找物体，请保持完整并稍作停顿；清晰定位后自动继续')
                self.set_status(self.msg('学习暂停：'+reason,
                    'Capture paused: keep object clear and complete; Esc to reselect if lost.'))
                return
            self.pending_box = decision.box
            if decision.action == 'accept':
                rx1,ry1,rx2,ry2 = self.pending_box
                add_reference_view(self.collect_reference_views,self.frame[ry1:ry2,rx1:rx2])
                # Repeated observations may add augmentation variants, but their
                # pixels always come from the last independently accepted frame.
                self._collect_aug_frame = self.frame.copy()
                self._collect_aug_box = self.pending_box
            self.set_status(self.t("collecting"))
            capture_frame = self._collect_aug_frame
            b = self._jitter_box(self._collect_aug_box)
            crop_full = crop_resize(capture_frame, b, self.in_size)
            crops = [crop_full]
            bw, bh = b[2] - b[0], b[3] - b[1]
            if min(bw, bh) >= 24:   # 中心裁剪增强：突出物体、抑制背景
                ix, iy = int(bw * 0.175), int(bh * 0.175)
                crop_ctr = crop_resize(capture_frame, (b[0] + ix, b[1] + iy, b[2] - ix, b[3] - iy),
                                       self.in_size)
                # 纹理感知：中心裁剪若已无纹理(纯色)，会成为'和任何均匀背景都像'的
                # 垃圾样本（枕头/墙面误识别的元凶），直接丢弃
                var_f = cv2.Laplacian(cv2.cvtColor(crop_full, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
                var_c = cv2.Laplacian(cv2.cvtColor(crop_ctr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
                if var_c >= max(12.0, 0.5 * var_f):
                    crops.append(crop_ctr)
            # 正样本增强只使用框内图像。扩大原画面取图范围会将框外的
            # 手、衣服和房间背景登记成物体，不能用它模拟物体举近。
            # 小角度旋转增强(±12°)：拍摄/拿取角度一变特征就漂移是 32x32
            # 小特征的老大难——注册时把角度包络直接学进去
            roi0 = capture_frame[b[1]:b[3], b[0]:b[2]]
            if roi0.size and min(bw, bh) >= 24:
                zin = self.in_size[0] if isinstance(self.in_size, (tuple, list)) else self.in_size
                rc = (roi0.shape[1] / 2.0, roi0.shape[0] / 2.0)
                for ang in (-12, 12):
                    M = cv2.getRotationMatrix2D(rc, ang, 1.0)
                    rot = cv2.warpAffine(roi0, M, (roi0.shape[1], roi0.shape[0]),
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_REFLECT_101)
                    crops.append(illum_norm(cv2.resize(rot, (int(zin), int(zin)),
                                                       interpolation=cv2.INTER_AREA)))
                # 水平翻转增强(左右镜像): 拿取方向变化时的姿态覆盖
                fl = cv2.flip(roi0, 1)
                crops.append(illum_norm(cv2.resize(fl, (int(zin), int(zin)),
                                                   interpolation=cv2.INTER_AREA)))
            # 远距尺度增强(0.5x/0.7x)：把框内图先缩小再放大回输入尺寸, 模拟
            # '物体更小/更远'时的细节丢失——一次性学到远近不同尺度的特征,
            # 缓解远距离识别失效(推理端的 0.45/0.6 扫描档也配套覆盖小目标)。
            if roi0.size and min(bw, bh) >= 24:
                zin2 = self.in_size[0] if isinstance(self.in_size, (tuple, list)) else self.in_size
                base = cv2.resize(roi0, (int(zin2), int(zin2)), interpolation=cv2.INTER_AREA)
                for z in (0.5, 0.7):
                    side_z = max(4, int(zin2 * z))
                    sm = cv2.resize(base, (side_z, side_z), interpolation=cv2.INTER_AREA)
                    crops.append(illum_norm(cv2.resize(sm, (int(zin2), int(zin2)),
                                                       interpolation=cv2.INTER_LINEAR)))
            x = preprocess_batch(np.stack(crops))
            self.collect_feats.append(self.backbone(x))   # 框内原图与增强图的特征
            # 背景负样本：同时记录'框外周围环境'长什么样(每帧取两条最大环带)。
            # 识别时窗口必须比背景更像物体本身(背景门控)——注册框里
            # 混进背景时，光秃秃的背景/桌面就再也冒充不了物体
            strips = self._bg_strips(b)
            if strips:
                zin = self.in_size[0] if isinstance(self.in_size, (tuple, list)) else self.in_size
                patches = np.stack([
                    illum_norm(cv2.resize(capture_frame[s[1]:s[3], s[0]:s[2]],
                                          (int(zin), int(zin)),
                                          interpolation=cv2.INTER_AREA)) for s in strips])
                self.collect_bg_feats.extend(self.backbone(preprocess_batch(patches)))
                # 环带颜色直方图: 与框内直方图比对, 判断'框里到底是不是背景'
                # ——纯色物体与背景特征相近但颜色往往明显不同, 不能只看特征。
                for s in strips:
                    sr = capture_frame[s[1]:s[3], s[0]:s[2]]
                    if sr.size == 0:
                        continue
                    sh = cv2.cvtColor(crop_resize(capture_frame,s,self.in_size), cv2.COLOR_BGR2HSV)
                    sh_h = cv2.calcHist([sh], [0, 1, 2], None, [8, 8, 8],
                                        [0, 180, 0, 256, 0, 256])
                    cv2.normalize(sh_h, sh_h, norm_type=cv2.NORM_L1)
                    self.collect_bg_hists.append(sh_h.flatten().astype(np.float32))
            # 累积框内 HSV 颜色直方图（颜色门控用）
            roi = capture_frame[max(0, b[1]):b[3], max(0, b[0]):b[2]]
            if roi.size:
                # Use the SAME resized/normalized crop as inference. Applying
                # CLAHE before resizing changes saturation enough to reject
                # the learned object even when its current geometry matches.
                hsv = cv2.cvtColor(crop_full, cv2.COLOR_BGR2HSV)
                # 与识别端保持一致: 算直方图时同样剔除肤色像素(手拿物体时指头
                # 会稀释物体的颜色, 两端都不算它才公平)。框内几乎全是皮肤时
                # (理论上不该出现)退回不剔除, 避免得到空直方图。
                # Generic objects may have skin-like colours; keep all labelled pixels.
                m = None
                h = cv2.calcHist([hsv], [0, 1, 2], m, [8, 8, 8],
                                 [0, 180, 0, 256, 0, 256])
                cv2.normalize(h, h, norm_type=cv2.NORM_L1)
                self.collect_hists.append(h.flatten().astype(np.float32))
            self.collect_n += 1
            if self.collect_n >= self.args.shots_per_box:
                self._commit_registration()
        # B 模式：识别（手选框优先；否则整幅画面 + 自动定位框出物体）
        elif self.mode == "B" and self.interaction == "idle" and self.data.has_data():
            if self.manual_box is not None:
                crop = crop_resize(self.frame, self.manual_box, self.in_size)
                x = preprocess_batch(crop[None])
                feats = self.backbone(x)
                cur = self.clf.cosine_batch(feats, self.data)
                cur = self._hist_gate(crop[None], cur.copy())   # 与自动定位同一套门控
                cur = self._bg_gate(feats, cur)
                cur = self._generic_roi_gate(feats,cur,self.manual_box)
                self.manual_cosv = self._smooth_cosv(self.manual_cosv, cur[0])
            else:
                if self.auto_scan:
                    self._scan_tick()   # 跟踪每帧执行(轻量)，全局搜索内部再按节奏调度
                else:
                    self._wholeframe_tick()

    def _jitter_box(self, b):
        """仅在原注册框内随机内缩，避免平移时将框外背景带入正样本。"""
        x1, y1, x2, y2 = b
        w, h = x2 - x1, y2 - y1
        j = max(1, min(w, h) // 12)
        left, top, right, bottom = self.rng.integers(0, j + 1, size=4)
        return (x1 + int(left), y1 + int(top),
                x2 - int(right), y2 - int(bottom))

    def _bg_strips(self, b):
        """注册框四周的背景环带候选(上下左右各一条, 厚度=半框)，
        返回面积最大的两条(至少 16px 边长)。作为该类的'负样本'：
        注册时记住物体周围的环境，识别时窗口若与背景不够有区分度
        (不像物体、倒像背景)，就会被背景门控拒绝。"""
        x1, y1, x2, y2 = b
        tw, th = (x2 - x1) // 2, (y2 - y1) // 2
        cands = (
            (max(0, x1 - tw), max(0, y1 - th), min(self.W, x2 + tw), y1),          # 上
            (max(0, x1 - tw), y2, min(self.W, x2 + tw), min(self.H, y2 + th)),     # 下
            (max(0, x1 - tw), max(0, y1 - th), x1, min(self.H, y2 + th)),          # 左
            (x2, max(0, y1 - th), min(self.W, x2 + tw), min(self.H, y2 + th)),     # 右
        )
        outs = [s for s in cands if s[2] - s[0] >= 16 and s[3] - s[1] >= 16]
        outs.sort(key=lambda s: (s[2] - s[0]) * (s[3] - s[1]), reverse=True)
        return outs[:2]

    def _wholeframe_tick(self):
        """整幅画面持续识别（与原版 main.py 演示的 inference 行为一致），
        输出与各类样本的 Top-K 余弦相似度向量。同样过颜色/背景门控：
        空背景时全类被掩蔽，中央大字显示'无匹配'而不是硬猜一个类别"""
        crop = crop_resize(self.frame, (0, 0, self.W, self.H), self.in_size)
        x = preprocess_batch(crop[None])
        feats = self.backbone(x)
        cur = self.clf.cosine_batch(feats, self.data)
        cur = self._hist_gate(crop[None], cur.copy())
        cur = self._bg_gate(feats, cur)
        cur = self._generic_roi_gate(feats,cur,(0,0,self.W,self.H))
        self.frame_cosv = self._smooth_cosv(self.frame_cosv, cur[0])

    def _generic_roi_gate(self,features,scores,box):
        generic=[ci for ci in range(len(self.data.labels)) if self._generic_class(ci)]
        if not generic:return scores
        gated=self._neg_gate(features,scores.copy())
        for ci in generic:
            scores[0,ci]=gated[0,ci]
            if scores[0,ci]>=0 and not self._new_white_identity_ok(box,ci,float(scores[0,ci])):
                scores[0,ci]=-1.
        return scores

    def _smooth_cosv(self, prev, cur, alpha=0.3):
        """相似度向量的快速 EMA（alpha=0.3，比 PEFSL 演示的概率平滑更快）"""
        cur = np.asarray(cur, dtype=np.float64)
        if prev is None or len(prev) != len(cur):
            return cur
        return (1 - alpha) * prev + alpha * cur

    def active_cosv(self):
        """当前生效的余弦相似度向量（供图例概率条与大字显示）：
        手选框 > 自动定位 > 整幅画面"""
        if self.manual_box is not None and self.manual_cosv is not None:
            return self.manual_cosv
        if self.auto_scan and self.scan_cosv is not None:
            return self.scan_cosv
        if self.frame_cosv is not None:
            return self.frame_cosv
        return None

    def _commit_registration(self):
        if not self.collect_feats or not getattr(self,'collect_reference_views',[]):
            self.set_status(self.msg('尚无有效学习画面，请保持物体清晰完整',
                                     'No valid capture yet; keep the complete object clear.'))
            return
        label = self.pending_label
        feats = np.concatenate(self.collect_feats, axis=0)
        x1, y1, x2, y2 = self.pending_box
        frac = max(x2 - x1, y2 - y1) / float(min(self.W, self.H))
        # Optional operator limit. Size alone does not prove background mixing;
        # the default permits a complete large object instead of a title crop.
        max_frac = float(getattr(self.args, "max_reg_frac", 0.0) or 0.0)
        if max_frac > 0.0 and frac > max_frac:
            self.set_status(self.msg(
                f"学习框超过设定的尺寸上限({frac:.0%})；请调整距离或尺寸上限，保持物体完整",
                f"Registration exceeds configured size limit ({frac:.0%}); keep the complete object."))
            print(f"[learn] REFUSED label='{label}' frac={frac:.2f} > {max_frac:.2f}")
            self.interaction = "idle"
            self.pending_box = None
            self.collect_capture = None
            self._collect_aug_frame = None
            self._collect_aug_box = None
            self.collect_feats, self.collect_hists, self.collect_n, \
                self.collect_bg_feats, self.collect_bg_hists = [], [], 0, [], []
            return
        # 保存该类别的注册缩略图（图例显示用）
        reference_views = list(getattr(self,'collect_reference_views',[]))
        # Only accepted captures can become references. The last camera frame
        # may be a redundant, blurred or displaced observation.
        best = max(reference_views, key=lambda image: reference_sharpness(
            cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),(0,0,image.shape[1],image.shape[0])))
        thumb = cv2.resize(best, (64,64), interpolation=cv2.INTER_AREA)
        # HSV 颜色直方图（颜色门控用：低纹理物体主要靠颜色区分）
        hist = None
        if self.collect_hists:
            hist = np.mean(np.stack(self.collect_hists, axis=0), axis=0)
            s = float(hist.sum())
            if s > 0:
                hist = (hist / s).astype(np.float32)
            else:
                hist = None
        # 注册质量检查用未中心化的原始特征方向，仅拒绝近乎相同的
        # 框内/框外内容。首次注册不能减去本次物体均值：物体均值会变为
        # 零向量，无法计算有意义的余弦。此检查不替代识别期 _bg_gate。
        if self.collect_bg_feats:
            bgm = np.stack(self.collect_bg_feats, axis=0).mean(axis=0)
            mean0 = np.zeros(feats.shape[1], dtype=np.float32)
            bvec = feature_preprocess(bgm[None].astype(np.float32), mean0)[0]
            mvec = feature_preprocess(feats.mean(axis=0, keepdims=True).astype(np.float32),
                                      mean0)[0]
            bg_sim = float(bvec @ mvec)
        else:
            bg_sim = 0.0
        # 框内 vs 框外环带的颜色距离：距离大说明框里是'另一种颜色的东西',
        # 不是背景。纯色物体(白色涂改带等)在 32x32 特征空间里天然与平坦背景
        # 很像(bg_sim 偏高), 但只要颜色明显不同就不该判成'框进了背景'——
        # 只看特征会让所有纯色物体都被误拒(用户反馈的'总是被拒绝')。
        bg_color_d = 1.0
        if self.collect_bg_hists and hist is not None:
            bh = np.mean(np.stack(self.collect_bg_hists, axis=0), axis=0)
            bs = float(bh.sum())
            if bs > 0:
                bg_color_d = float(cv2.compareHist(
                    hist.astype(np.float32), (bh / bs).astype(np.float32),
                    cv2.HISTCMP_BHATTACHARYYA))
        context_conflict = bg_sim > 0.98 and bg_color_d < 0.28
        self.data.add_shots(label, feats, frac, thumb, hist, self.collect_bg_feats,
                            box_ar=(x2 - x1) / float(max(1, y2 - y1)),
                            shape=self._shape_desc((x1, y1, x2, y2)),
                            reference_views=reference_views,generic=True,
                            background_trusted=False)
        self._data_changed()
        total = self.data.total_shots(label)
        # 注册质量检查：给用户可操作的提示
        warns = []
        if context_conflict:
            warns.append("物体与周围外观接近，请确认框住完整物体，可补录不同背景或角度")
        if min(x2 - x1, y2 - y1) < 64:
            warns.append("物体太小/太远，建议举近摄像头")
        if frac > 0.5:
            warns.append("学习框较大，请确认覆盖完整物体并尽量减少手部遮挡")
        # 纹理量检查：深色/纯色无纹理物体在 32x32 特征下无法与阴影等区分
        reg_crop = crop_resize(self.frame, (x1, y1, x2, y2), self.in_size)
        tex = cv2.Laplacian(cv2.cvtColor(reg_crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        if tex < 20:
            warns.append("物体纹理较少，建议补录其他角度或背景，保持完整外形")
        if len(self.data.labels) >= 2:
            m = self.data.class_means()
            m = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-8)
            i = self.data.labels.index(label)
            hi = self.data.class_hist(i)
            for j in range(len(self.data.labels)):
                if j != i and float(m[i] @ m[j]) > 0.92:
                    extra = ""
                    hj = self.data.class_hist(j)
                    if hi is not None and hj is not None:
                        batt = cv2.compareHist(hi, hj, cv2.HISTCMP_BHATTACHARYYA)
                        extra = (f"(特征相似且颜色距离{batt:.2f}也小)" if batt < 0.25
                                 else "(但颜色差异明显，颜色门控可区分)")
                    warns.append(f"与'{self.data.labels[j]}'特征很像{extra}，"
                                 f"建议换易区分的物体或贴边重选")
                    break
        # 背景同化预警：特征像背景但颜色差得开 -> 只是提醒(纯色物体常见)；
        # 特征像背景且颜色也像 -> 上面已经直接拒绝了。
        if bg_sim > 0.98:
            warns.append(f"框内特征与周围背景较接近(特征{bg_sim:.2f}/颜色距离"
                         f"{bg_color_d:.2f})，纯色物体常见；可靠度略低，"
                         f"建议贴边框紧或换个角度重注册")
        capture = getattr(self,'collect_capture',None)
        unique = getattr(capture,'accepted',len(self.collect_feats))
        self.last_registration = dict(label=label, observations=self.collect_n,
            accepted=unique, augmented=len(feats), references=len(reference_views),
            duplicates=getattr(capture,'duplicates',0), skipped=getattr(capture,'skipped',0))
        if unique < 3:
            warns.append("有效外观较少，可用同一标签补录不同真实角度；增强不等于新视角")
        zh = (f"已学习 '{label}'：观察 {self.collect_n} 帧，有效采样 {unique} 帧，"
              f"增强后本次 {len(feats)} 个、该类共 {total} 个特征样本；"
              f"背景负样本 {len(self.collect_bg_feats)} 个")
        en = (f"Learned '{label}': {self.collect_n} observations, {unique} accepted, "
              f"{len(feats)} augmented features added; {total} total")
        if warns:
            zh += "  [!] " + "；".join(warns)
            en += "  [!] " + "; ".join(warns)
        self.set_status(self.msg(zh, en))
        print(f"[learn] label='{label}' +{len(feats)} feats (total {total}) frac={frac:.2f}"
              + (f" bg={len(self.collect_bg_feats)}" if self.collect_bg_feats else "")
              + (f" warns={warns}" if warns else ""))
        self.interaction = "idle"
        self.collect_capture = None
        self._collect_aug_frame = None
        self._collect_aug_box = None
        self.pending_box = None
        self.collect_feats, self.collect_hists, self.collect_n, \
            self.collect_bg_feats, self.collect_bg_hists = [], [], 0, [], []

    def _generic_colour_references(self,ci):
        """Match colour after the SAME resize/illumination pipeline as query.

        Historical histograms were normalized at camera ROI resolution, while
        queries use backbone resolution. CLAHE changes saturation with image
        size. Rebuild only this runtime gate from saved raw learned views;
        saved features, prototypes and archives remain unchanged.
        """
        key=(self._search_epoch,ci,tuple(self.in_size))
        cache=getattr(self,'_generic_colour_reference_cache',{})
        if key not in cache:
            references=[]
            for view in self.data.class_reference_views(ci):
                h,w=view.shape[:2]
                crop=crop_resize(view,(0,0,w,h),self.in_size)
                hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
                hist=cv2.calcHist([hsv],[0,1,2],None,[8,8,8],[0,180,0,256,0,256])
                hist/=max(float(hist.sum()),1.)
                references.append(hist.flatten())
            cache[key]=references or [self.data.class_hist(ci)]
            self._generic_colour_reference_cache=cache
        return cache[key]

    def _hist_gate(self, crops, coss, white_sv=False, generic_compensation=()):
        """颜色门控（逐类掩蔽）：对每个窗口分别与每个类别的注册直方图比对
        Bhattacharyya 距离，颜色不符的类被屏蔽(-1)。这样类别判定在颜色
        可行的类里选相似度最高的——两个类特征相近时（如都是浅色盒），
        颜色是唯一判别维度，物体2的窗口会屏蔽掉类1、正确标为类2。
        全部类都被屏蔽 = 该窗口不是任何已学物体（无匹配）。
        歧义拒绝：两个类都通过颜色门且相似度差距 < 0.04 时（特征+颜色都无法
        区分的窗口），拒绝猜测、整窗判无匹配——宁可不识别，不标错类别。
        无直方图数据(旧版保存)时自动跳过。crops: (N,32,32,3)。"""
        self._hist_reasons = np.full(coss.shape, "通过", dtype=object)
        if not self.data.labels or coss.size == 0:
            return coss
        hists = [self.data.class_hist(ci) for ci in range(len(self.data.labels))]
        if all(h is None for h in hists):
            return coss
        generic = {ci for ci in range(len(hists)) if self._generic_class(ci)}
        legacy = [ci for ci in range(len(hists)) if ci not in generic]
        for j,crop in enumerate(crops):
            # All learned views inspect the same current pixels. Prepare this
            # unmasked query once; never retain it across crops or frames.
            prepared=generic_colour_query(crop) if generic else None
            for ci in generic:
                if not any(generic_colour_supported(crop,h,self.args.scan_hist_th,
                           compensate=ci in generic_compensation,prepared=prepared)
                           for h in self._generic_colour_references(ci)):
                    coss[j,ci] = -1.
                    self._hist_reasons[j,ci] = '颜色与完整学习区域不符'
        # 各类的 饱和度S / 亮度V 单通道边缘分布(8桶)。三维直方图在'只有一个
        # 通道不同'时距离会被另两个一致的通道稀释——比如格子衬衫的白块(亮、
        # 但混着大片黑格)与白色涂改带(亮、干净)在色相/饱和度上一致, 只有
        # 亮度分布不同, 三维距离就偏小、拦不住。拆成单通道分布后立刻分开,
        # 专门治'白格子被认成白色物体'。
        ref_marg = []
        for h in hists:
            if h is None:
                ref_marg.append(None)
                continue
            t3 = np.asarray(h, np.float32).reshape(8, 8, 8)
            t3 = t3 / max(float(t3.sum()), 1e-9)
            v_m = t3.sum(axis=(0, 1))                                  # V 边缘
            s_m = t3.sum(axis=(0, 2))                                  # S 边缘
            ref_marg.append((s_m.astype(np.float32),                   # S 边缘
                             v_m.astype(np.float32),
                             float(v_m[:2].sum()),                     # 暗像素(V<64)占比
                             float(s_m[0])))                           # 极低饱和占比(白/灰)
        # 白/灰类的新锁定门槛(与 _scan_tick 里 th_ci += 0.12 那条保持一致)：
        # 强匹配豁免要用"该类自己的门槛"当参照, 而不是一个固定的 0.85。
        whites = self._white_flags()
        th_lock_white = float(self.args.scan_cos_th) + 0.02 + 0.12
        for j in range(crops.shape[0]):
            pre = coss[j].copy()      # 颜色门之前的分数(强匹配豁免要用)
            hsv = cv2.cvtColor(crops[j], cv2.COLOR_BGR2HSV)
            # 算颜色直方图时**先剔除肤色像素**: 手拿物体时指头常占窗口三四成,
            # 不剔除会把物体本身的颜色稀释掉(真物体颜色对不上->漏检)。反过来,
            # 脸/脖子这种几乎全是皮肤的窗口, 剔除后剩下的像素既少又杂(眼镜框、
            # 头发边), 直方图和'物体的颜色'天然对不上 -> 被颜色门拒掉。
            # 用一条'肤色占比'是分不开手和脸的(两者重叠), 这样分开才靠谱。
            # 肤色判据用 HSV(与 _skin_ratio 一致): 色相 0~30 且饱和度 >=12 ——
            # 中性灰/白饱和度≈0 被排除, 被拍白的皮肤仍带红黄底色能认出来。
            skin = ((hsv[..., 0] <= 30) & (hsv[..., 1] >= 12)
                    & (hsv[..., 1] <= 180) & (hsv[..., 2] >= 50))
            # 非皮肤像素不足 40% 就判'不是物体'(原 25% 太松): 手腕/脸/脖子
            # 这类大块皮肤窗口, 剔掉皮肤后只剩零星边缘, 25% 的线放它过关,
            # 剩下那点边缘再蹭上白色模型 -> 框就锁在手腕上了。
            # 手拿物体时指头通常占 3~4 成, 非皮肤仍有 60%+, 不受影响。
            if (skin.size - int(skin.sum())) < 0.40 * skin.size:
                coss[j, legacy] = -1.0
                self._hist_reasons[j, legacy] = f"肤色占比{skin.mean():.0%}"
                continue
            mask = (~skin).astype(np.uint8) * 255
            hw = cv2.calcHist([hsv], [0, 1, 2], mask, [8, 8, 8],
                              [0, 180, 0, 256, 0, 256])
            cv2.normalize(hw, hw, norm_type=cv2.NORM_L1)
            hw = hw.flatten().astype(np.float32)
            t3 = hw.reshape(8, 8, 8)
            m_s = t3.sum(axis=(0, 2)).astype(np.float32)
            m_v = t3.sum(axis=(0, 1)).astype(np.float32)
            # 低饱和度窗口的 HSV 色调分量基本是噪声, 直方图每帧不一样, 用常规
            # 阈值会把真物体整类掩蔽(0%) -> 需要放宽。
            # 但宽松规则**只能给'明亮的低饱和窗口'(白/浅灰物体)**: 黑发、深色
            # 木柜、暗背景同样饱和度低, 若一并放宽, 它们与明亮物体类颜色差极大
            # 却照样通过颜色门——这就是'头发/暗处被认成涂改带或书本'的根源。
            # 所以用亮度 V 区分: 只有又亮又低饱和的窗口才放宽。
            s_mean = float(hsv[..., 1].mean())
            v_mean = float(hsv[..., 2].mean())
            th_use = 0.78 if (s_mean < 35 and v_mean > 120) else self.args.scan_hist_th
            m_th = self.args.scan_marg_th
            for ci, h in enumerate(hists):
                if h is None or ci in generic:
                    continue
                # 强匹配豁免: 特征相似度已经很高(>=0.85)时, 颜色门不再否决。
                # 道理: 颜色门是给'特征分不出高下'时用的; 87% 的绝对余弦已经
                # 极强。而物体贴近镜头时框内只剩封面局部(大标题字+底色), 内容
                # 占比与注册时差很多, 亮度分布还随曝光漂——实测书贴脸时颜色
                # 距离 0.90/亮度 0.60 被判掉, 明明 87% 却'无匹配'。
                if pre[ci] >= 0.85 and not (white_sv and whites[ci]):
                    continue
                # 遮挡容忍: 物体被别的东西挡住一部分时, 窗口里会混进遮挡物, 颜色
                # 分布整体必然偏离 -> 只看'对称距离'会误杀。这里额外用**非对称
                # 重合度**(直方图交集 sum(min))兜底: 只要'物体本来的那些颜色'仍有
                # 相当比例出现在窗口里, 就认为物体还在(只是被挡了一部分), 放行。
                # 实测约能容忍一半被遮; 完全不相干的东西(红柜子 vs 绿书)重合度
                # 接近 0, 照样被拦。
                rs, rv, rdark, rwhite = ref_marg[ci]
                use_sv = bool(white_sv and rwhite > 0.60)
                if use_sv:
                    # Hue is unstable near zero saturation. Keep joint S/V
                    # statistics and the dark-pixel check for fitted white objects.
                    color_query = t3.sum(axis=0).astype(np.float32).flatten()
                    color_ref = np.asarray(h, np.float32).reshape(8,8,8).sum(axis=0).flatten()
                else:
                    color_query, color_ref = hw, h
                ov3 = float(np.minimum(color_query, color_ref).sum())
                # 放行规则: '距离够近' 或 '重合度够高' 任一条满足即通过(容忍贴近、
                # 被遮挡、以及白色物体在强光下的颜色漂移)。
                # 注: 曾试过对白/灰类改成'两条都必须满足'(重合度≥0.55), 结果
                # **把真涂改带也拦掉了**——实测同一支涂改带换个举法: 三维距离
                # 0.51~0.62、重合度只有 0.22~0.48; 而误检的耳朵是 0.66/0.27。
                # 两者在两个维度上几乎重合, 加严只会两败俱伤, 所以回到宽松版。
                # 白/灰类额外收一道颜色距离(0.68→0.65): 实测同一支涂改带的三维
                # 颜色距离是 0.51~0.62, 而误检的眼睛/眼镜那块约 0.66 —— 两个维度
                # (距离/重合度)里, **只有距离这一维有一点分辨力**, 就在这里收紧。
                # 注意用 min(而不是直接覆盖): 明亮低饱和窗口本来有 0.78 的放宽,
                # 白/灰类的窗口正好都是那种, 直接覆盖会把真物体一起误伤。
                cth = min(th_use, 0.65) if rwhite > 0.60 else th_use
                if use_sv:
                    cth = min(th_use, 0.65)
                # 【白/灰类的**相对豁免线**(本轮新增)】
                #   原来只有一条固定线 pre>=0.85 才豁免颜色门。但白/灰类的新锁定
                #   门槛本身被抬到 0.70, 而实测(2026-10-13 diag 两帧)真涂改带举近时
                #   原始分只有 0.75~0.82 —— **低于 0.85, 拿不到豁免**, 于是被直方图
                #   距离整类掩蔽, 屏上显示"无匹配 0%"却明明握着物体:
                #     09:32:15  1:82/69/69  (0.82 的窗口被掩蔽, 剩下 0.69<0.70 被拒)
                #     09:32:19  1:75/-100/-100 (0.75 的窗口被掩蔽 -> 全类 -100)
                #   而同一批实测里, 衬衫领口那个误检是 76/76/76 —— 颜色门**本来
                #   就没拦住它**。可见对白/灰类, 这条直方图距离判据弊大于利。
                #   故改成: 原始分已超过"该类自己的落锁门槛"0.05 以上 -> 不再用
                #   直方图距离否决。**只豁免这一条**; 下面的暗像素判据照旧生效
                #   (它是结构性的, 正是拦'格子衬衫白块'的那条, 不能一起放掉)。
                strong_w = not use_sv and whites[ci] and pre[ci] >= th_lock_white + 0.05
                color_dist = float(cv2.compareHist(color_query, color_ref, cv2.HISTCMP_BHATTACHARYYA))
                if (not strong_w) and ov3 < 0.35 and color_dist > cth:
                    coss[j, ci] = -1.0
                    self._hist_reasons[j, ci] = f"{'SV' if use_sv else 'HSV'}颜色距离{color_dist:.2f}>{cth:.2f}"
                    continue
                # 注: 原来还有'饱和度/亮度单通道分布'两道检查, 已删除——它们
                # 对框内内容占比和曝光极其敏感(书拿近一点、封面文字占比一变、
                # 光线一变就漂), 反复把真物体整类掩蔽('书放那么久识别不出来')。
                # 现在靠 三维直方图距离 + 重合度 + 暗像素占比 三条, 都不那么脆。
                # 暗像素占比: 该类注册时几乎没有暗像素(白色物体=0%), 窗口里却混着
                # 黑格/深色纹理 -> 不是这个物体。
                # 实测: 涂改带那一类注册时暗像素 0.0%, 而格子衬衫上被误锁的那块
                # 暗像素占 16.5% -> 判定线设 0.12 刚好拦住它, 又给真物体留足余量
                # (白色物体正常光照下暗像素≈0)。这是'白格子被认成白色涂改带'
                # 最直接的一条判据。
                if float(m_v[:2].sum()) - rdark > 0.12:
                    coss[j, ci] = -1.0
                    self._hist_reasons[j, ci] = "暗像素偏多"
            valid = coss[j][coss[j] > 0]
            if len(valid) >= 2:
                srt = np.sort(valid)[::-1]
                if srt[0] - srt[1] < 0.04:   # 歧义: 两个类都无法区分，拒绝猜测
                    coss[j, :] = -1.0
                    self._hist_reasons[j, :] = "类别歧义"
        return coss

    def _bg_gate(self, feats, coss, margin=0.08):
        """背景负样本门控：窗口与某类'注册时框外背景'的余弦相似度接近甚至
        超过其与该类物体的相似度(差 < margin)时，掩蔽该类(-1)。
        治本：注册框混入背景后，空场景/桌面背景的窗口与背景样本几乎一样，
        即使与物体类相似度冒头(0.6x)也会被这里拒绝——'光秃秃的背景识别出
        物体'从此走不通。真物体窗口与背景差异明显，不受影响。
        在与 cosine_batch 一致的均值减法+单位球空间里计算。
        feats: (N,D) 原始 backbone 特征; coss: (N,C)，就地修改。"""
        C = len(self.data.labels)
        if coss is None or getattr(coss, "size", 0) == 0 or feats.ndim != 2:
            return coss
        if coss.ndim != 2 or coss.shape[0] != feats.shape[0] or coss.shape[1] != C:
            return coss
        bgs = [self.data.class_bg(ci) for ci in range(C)]
        if all(b is None for b in bgs):
            return coss                       # 旧数据无背景样本，跳过
        mean = self.data.mean_feature()
        F = feature_preprocess(np.asarray(feats, dtype=np.float32), mean)   # (N,D)
        for ci in range(C):
            if bgs[ci] is None:
                continue
            base=np.zeros_like(mean) if self._generic_class(ci) else mean
            B = feature_preprocess(bgs[ci][None].astype(np.float32), base)[0]
            query=feature_preprocess(np.asarray(feats,np.float32),base) if self._generic_class(ci) else F
            cosbg = query @ B
            bad = (coss[:, ci] - cosbg) < (.005 if self._generic_class(ci) else margin)
            if bad.any():
                coss[bad, ci] = -1.0
        return coss

    def _cs_gate_ok(self, b, cs_th=0.16):
        """中心-周围对比门控（仅用于新锁定/切换）：
        窗口与其外圈背景的 HSV 颜色直方图 Bhattacharyya 距离 < cs_th 说明
        '窗口内容和周围环境几乎一样'——桌面/墙面/床单等大片均匀背景的典型
        特征，即使特征相似度冒头也拒绝锁定。真正的物体应与背景有明显反差。
        外圈过小(窗口几乎铺满画面/贴边)时无法判断，放行。"""
        x1, y1, x2, y2 = b
        w, h = x2 - x1, y2 - y1
        if w < 16 or h < 16:
            return True
        mx1, my1 = max(0, x1 - w // 2), max(0, y1 - h // 2)
        mx2, my2 = min(self.W, x2 + w // 2), min(self.H, y2 + h // 2)
        rw, rh = mx2 - mx1, my2 - my1
        ring_area = rw * rh - w * h
        if ring_area < w * h // 2:      # 外圈面积不足窗口一半，无法可靠判断
            return True
        ring = self.frame[my1:my2, mx1:mx2]
        mask = np.full(ring.shape[:2], 255, np.uint8)
        mask[y1 - my1:y2 - my1, x1 - mx1:x2 - mx1] = 0
        hsv = cv2.cvtColor(ring, cv2.COLOR_BGR2HSV)
        h_ring = cv2.calcHist([hsv], [0, 1, 2], mask, [8, 8, 8],
                              [0, 180, 0, 256, 0, 256])
        cv2.normalize(h_ring, h_ring, norm_type=cv2.NORM_L1)
        roi = self.frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h_win = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8],
                             [0, 180, 0, 256, 0, 256])
        cv2.normalize(h_win, h_win, norm_type=cv2.NORM_L1)
        d = float(cv2.compareHist(h_win.flatten().astype(np.float32),
                                  h_ring.flatten().astype(np.float32),
                                  cv2.HISTCMP_BHATTACHARYYA))
        return d >= cs_th

    def _shape_desc(self, b):
        """窗口的形状/纹理描述子 (边缘密度, 灰度对比度)——弱化颜色、
        强化轮廓与结构判据的基础量。"""
        x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
        roi = self.frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        if roi.shape[0] < 8 or roi.shape[1] < 8:
            return np.zeros(2, np.float32)
        h, w = roi.shape[:2]
        if max(h, w) > 96:
            s = 96.0 / max(h, w)
            roi = cv2.resize(roi, (max(8, int(w * s)), max(8, int(h * s))),
                             interpolation=cv2.INTER_AREA)
        # 先做暗光增强再取轮廓: Canny 用的是绝对阈值(40/120), 暗光下几乎抓不到
        # 边缘、灰度对比度也整体塌下去 -> 形状门控会把真物体误判成'纯色背景'。
        # 归一光照后注册/识别两侧都在同一亮度空间里比较。
        g = cv2.cvtColor(illum_norm(roi), cv2.COLOR_BGR2GRAY)
        e = cv2.Canny(g, 40, 120)   # 阈值放宽: 暗光下边缘弱, 原 60/160 会漏检
        return np.array([float(e.mean()) / 255.0, float(g.std())], np.float32)

    def _shape_gate_ok(self, b, ci, white=False):
        """形状/轮廓门控：窗口的边缘密度与灰度对比度必须与该类注册窗口
        相当。纯色/平坦背景区域(边缘密度、对比度极低)即使颜色或特征接近
        也拒绝锁定——'弱化颜色权重、优先按轮廓与几何结构匹配'的落点。
        white=True(白/灰类) 时还可启用**灰度对比度上限**(--white-max-contrast):
        实测误检的"白衣服/白桌面块"灰度对比度是注册值的 1.8 倍(42.8~45.0 vs
        23.9), 而边密度只有 0.88~0.91 倍 —— 对比度这一维比边密度更分得开。
        默认关闭(0), 因为上限偏紧、可能误伤对比度高的姿态。"""
        ref = self.data.class_shape(ci)
        if ref is None:
            return True
        sd = self._shape_desc(b)
        ed, ct = float(sd[0]), float(sd[1])
        red, rct = float(ref[0]), float(ref[1])
        if red > 0.010 and ed < 0.35 * red:
            return False       # 轮廓/纹理远少于注册物体: 疑似纯色背景
        if rct > 8.0 and ct < 0.35 * rct:
            return False       # 灰度对比度远低: 平坦区域
        max_r = float(getattr(self.args, "white_max_contrast", 0.0) or 0.0)
        if white and max_r > 0.0 and rct > 8.0 and ct > max_r * rct:
            return False       # 对比度远高于注册物体: 疑似织物/桌面纹理块
        return True

    def _generic_class(self, ci):
        return bool(getattr(self.data,'is_generic',lambda i:False)(ci))

    def _bright_localizer_flags(self):
        # Bright/transparent endpoint heuristics are a compatibility policy
        # for historical archives, never the default for a new white object.
        return [white and not self._generic_class(ci)
                for ci,white in enumerate(self._white_flags())]

    def _white_flags(self):
        """每一类是否'白/灰类': 注册颜色直方图里**极低饱和像素占比 > 0.60**。
        这类颜色模型太泛(白墙/白纸/皮肤高光/眼镜反光都是'亮+低饱和'), 所以
        '绝对负样本门'只对它们启用, 不影响有颜色的类。无直方图 -> False。"""
        out = []
        for i in range(len(self.data.labels)):
            h = self.data.class_hist(i)
            if h is None:
                out.append(False)
                continue
            t3 = np.asarray(h, np.float32).reshape(8, 8, 8)
            t3 = t3 / max(float(t3.sum()), 1e-9)
            out.append(float(t3.sum(axis=(0, 2))[0]) > 0.60)
        return out

    def _neg_gate(self, feats, coss, margin=0.02):
        """负样本门控(局部化)：只有当窗口'比该类物体更像用户标记的负样本'
        (相似度高出 margin)时才掩蔽该类，而不是'有点像就压分'——保证
        标记误检背景时不会顺带把与背景有几分相似的正样本也误伤。"""
        if self.data.neg_feats is None or len(self.data.neg_feats) == 0:
            return coss
        if coss is None or getattr(coss, "size", 0) == 0 or feats.ndim != 2:
            return coss
        mean = self.data.mean_feature()
        if mean is None:
            return coss
        F = feature_preprocess(np.asarray(feats, np.float32), mean)
        generic=[ci for ci in range(coss.shape[1]) if self._generic_class(ci)]
        generic_scores=coss[:,generic].copy()
        N = feature_preprocess(np.asarray(self.data.neg_feats, np.float32), mean)
        cosn = (F @ N.T).max(axis=1)          # (N,) 与最像的那个负样本
        bad = (cosn[:, None] - coss) > margin  # 负样本明显更像 -> 才掩蔽
        # 【绝对规则(只对白/灰类, 本轮新增)】
        #   相对规则对"白物体 vs 白背景/白衣"**无效**: 窗口"像本类"和"像负样本"
        #   的程度本就相当, `cosn - coss` 永远不超过 margin, 于是标了负样本也不掩蔽
        #   (用户实测: 标了 6 处、仍然锁在衣领上)。这类白/灰物体改用**绝对门**:
        #   窗口与任一负样本的相似度 >= --neg-abs-th 就掩蔽本类。
        #   之所以敢用绝对门: 负样本标的是**具体的误检位置**, 同一处的窗口相似度
        #   接近 1 会稳稳过线; 而真物体就算同是白色, 也通常达不到这个高度。
        #   --neg-abs-th 就是在这两者之间取界(默认 0.90; 设 0 关闭)。
        abs_th = float(getattr(self.args, "neg_abs_th", 0.0) or 0.0)
        if abs_th > 0.0:
            white = self._white_flags()
            if len(white) == coss.shape[1]:
                hit = cosn >= abs_th
                if hit.any():
                    for ci in range(coss.shape[1]):
                        if white[ci]:
                            bad[hit, ci] = True
        if bad.any():
            coss[bad] = -1.0
        if generic:
            origin=np.zeros_like(mean)
            raw_query=feature_preprocess(np.asarray(feats,np.float32),origin)
            raw_negative=feature_preprocess(np.asarray(self.data.neg_feats,np.float32),origin)
            similarity=(raw_query@raw_negative.T).max(axis=1)
            generic_scores[(similarity[:,None]-generic_scores)>=-.005]=-1.
            coss[:,generic]=generic_scores
        return coss

    def _shape_weight(self, boxes, coss):
        """用目标独有结构(边缘密度/灰度对比度)对特征分做**轻微**校正：
        coss *= (0.85 + 0.15 × 结构符合度)，最多只扣 15%。
        结构描述子只有两个数, 会随光照、远近、框内背景比例波动(拿近/换角度
        就变), 只适合当'微调', 绝不能一票否决——实测它把书从 66 分压到 43 分
        直接漏检('书放那么久都识别不出来')。真正把关的是颜色门/中心-周围门/
        形状下限门, 结构权重只做最后的微调。"""
        if not boxes or coss is None or coss.size == 0:
            return coss
        mxc = coss.max(axis=1)
        lo = mxc < 0.35                     # 低分窗口不参与形状计算(省 Canny),
        coss[lo] *= 0.85                    # 直接按最低结构权重压分
        hi = np.where(~lo)[0]
        if len(hi) == 0:
            return coss
        sds = np.stack([self._shape_desc(boxes[k]) for k in hi]).astype(np.float32)
        ed, ct = sds[:, 0], sds[:, 1]
        for ci in range(len(self.data.labels)):
            ref = self.data.class_shape(ci)
            if ref is None:
                continue
            red = max(float(ref[0]), 1e-6)
            rct = max(float(ref[1]), 1e-6)
            r1 = np.minimum(ed, red) / np.maximum(ed, red)
            r2 = np.minimum(ct, rct) / np.maximum(ct, rct)
            # 容差带: 比值 >= 0.6 视为'结构相符'(不扣分), 更差才逐级扣
            f1 = np.clip((r1 - 0.25) / 0.35, 0.0, 1.0)
            f2 = np.clip((r2 - 0.25) / 0.35, 0.0, 1.0)
            sf = 0.5 * f1 + 0.5 * f2
            coss[hi, ci] = coss[hi, ci] * (0.85 + 0.15 * sf)
        return coss

    def _skin_ratio(self, b):
        """窗口内肤色像素占比 (YCbCr 简易肤色检测)。脸/额头/下巴/手/脖子等
        人体局部在此被识别并直接排除, 不进入目标特征匹配。
        关键: 先做暗光增强再检测——原色域范围(Cr133~173)是按正常光照定的,
        弱光下皮肤变暗、Cr/Cb 整体偏移, 不增强就会漏检(额头/脸被误认成物体)。"""
        x1, y1 = max(0, int(b[0])), max(0, int(b[1]))
        x2, y2 = min(self.W, int(b[2])), min(self.H, int(b[3]))
        roi = self.frame[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0
        # 先在缩略图上检测: 肤色占比不受缩放影响, 但 CLAHE/cvtColor 的开销
        # 降一个量级——这个函数现在每帧(跟踪期)每个目标都要调用一次, 大窗口
        # 直接跑 illum_norm 是实打实的帧率开销。
        rh, rw = roi.shape[:2]
        if max(rh, rw) > 96:
            s = 96.0 / max(rh, rw)
            roi = cv2.resize(roi, (max(8, int(rw * s)), max(8, int(rh * s))),
                             interpolation=cv2.INTER_AREA)
        roi = illum_norm(roi)      # 暗光增强: 提升弱光下肤色检测可靠性
        # 肤色判据用 HSV(不再用 YCrCb 的 Cr/Cb 阈值):
        #   皮肤(包括强光下被拍白的)都带一层红黄底色 -> 色相落在 0~30(即
        #   0~60°, 覆盖红/橙/黄), 且饱和度还有 12 以上;
        #   中性灰/白饱和度≈0(色相是噪声) -> 直接排除。
        #   Cr/Cb 阈值判据两头都踩过坑: 收紧要漏掉被拍白的皮肤(脸/手腕当成
        #   普通亮块去匹配白色物体), 放宽会把白色物体本身判成皮肤(白色物体
        #   永远认不出来)。色相+饱和度这个组合稳得多。
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hh_, ss_, vv_ = hsv_roi[..., 0], hsv_roi[..., 1], hsv_roi[..., 2]
        skin = (hh_ <= 30) & (ss_ >= 12) & (ss_ <= 180) & (vv_ >= 50)
        return float(skin.mean())

    def _geom_gate_ok(self, b, ci, fracs):
        """前置几何约束：候选面积必须与该类注册尺寸同量级(0.1~8.0 倍)。
        上限过滤头发/脸/衣物等大面积区域(耳机不会大面积出现); 下限只挡
        噪点级碎块。范围放宽到 8 倍是为了不误杀'举近'(面积可达 5-8 倍)。"""
        area = float((b[2] - b[0]) * (b[3] - b[1]))
        ref_side = float(fracs[ci]) * min(self.W, self.H)
        ref_area = ref_side * ref_side
        if ref_area <= 0:
            return True
        r = area / ref_area
        # 下限一路放到 0.02: 物体离得远时画面里可能只有注册尺寸的 1/5~1/6
        # 面积(实测涂改带最小扫描档就落在这个量级), 原来的下限直接把远距离
        # 目标挡在门外('离得很远就没办法识别')。上限 20 倍不变(举近的余量)。
        # 放这么低也安全: 这么小的窗口特征模糊, 分不够自然会被门槛筛掉。
        return 0.02 <= r <= 20.0

    def _refine_borderline_localized_views(self, box, ci, score, rows):
        """Resolve boundary sampling uncertainty without lowering the lock line."""
        lock=self.args.scan_cos_th+.02
        if not lock-.03<=score<lock or not self.data.class_visual_refs(ci):
            return score,rows,None
        if not self._localized_geometry_ok(box,ci) or not self._shape_gate_ok(box,ci,True):
            return score,rows,None
        reference=self._registered_appearance_similarity(box,ci)
        if reference<.62:
            return score,rows,None
        x1,y1,x2,y2=box
        views=[]
        for inset in (.03,.05):
            dx,dy=int((x2-x1)*inset),int((y2-y1)*inset)
            views.append((x1+dx,y1+dy,x2-dx,y2-dy))
        crops=np.stack([crop_resize(self.frame,b,self.in_size) for b in views])
        features=self.backbone(preprocess_batch(crops))
        raw=self.clf.cosine_batch(features,self.data)
        color=self._hist_gate(crops,raw.copy(),white_sv=True)
        background=self._bg_gate(features,color.copy())
        negative=self._neg_gate(features,background.copy())
        combined=[np.concatenate((old,extra[:,ci])) for old,extra in
                  zip(rows,(raw,color,background,negative))]
        qualified=combined[-1][np.isfinite(combined[-1]) & (combined[-1]>=lock)]
        if len(qualified)<2:
            return score,rows,None
        pair=np.sort(qualified)[-2:]
        refined=float(min(pair.mean(),pair[0]+.03))
        return refined,combined,dict(base_score=round(score,3),
            registered_appearance=round(reference,3),view_insets=[0.,.04,.08,.03,.05],
            qualifying_views=int(len(qualified)))

    def _score_localized_boxes(self, boxes):
        # Search refinement, candidate ranking and final admission often ask
        # for the same crop. Reuse only within the SAME captured frame and
        # learning/background epoch; tracking still evaluates every new frame.
        key=(self.frame_idx,getattr(self,'_capture_time',None),self._search_epoch)
        if getattr(self,'_localized_score_cache_key',None)!=key:
            self._localized_score_cache_key=key
            self._localized_score_cache={}
            self._localized_score_contexts={}
        boxes=[tuple(map(int,box)) for box in boxes]
        contexts=getattr(self,'_localized_score_contexts',{})
        self._localized_score_contexts=contexts
        pattern_current=getattr(self,'_pattern_cache_key',None)==(self.frame_idx,self._search_epoch)
        # Scores depend on current identity evidence (colour compensation and
        # aligned views). Keep strong references so object-id reuse cannot
        # make a newly published proof look like the cached context.
        current={box:tuple(getattr(self,'_pattern_cache',{}).get((ci,box)) if pattern_current else None
                    for ci in range(len(self.data.labels))) for box in dict.fromkeys(boxes)}
        missing=[box for box,evidence in current.items() if box not in self._localized_score_cache
                 or box not in contexts or len(contexts[box])!=len(evidence)
                 or any(old is not new for old,new in zip(contexts[box],evidence))]
        if missing:
            scores=self._compute_localized_boxes(missing)
            for box,row,evidence in zip(missing,scores,self._last_localized_evidence):
                self._localized_score_cache[box]=(row.copy(),copy.deepcopy(evidence))
                # Capture dependencies from BEFORE computation. If computing
                # the score itself discovers evidence, the next call rechecks.
                contexts[box]=current[box]
        self._last_localized_evidence=[copy.deepcopy(self._localized_score_cache[box][1]) for box in boxes]
        return np.stack([self._localized_score_cache[box][0] for box in boxes]) \
            if boxes else np.empty((0,len(self.data.labels)),dtype=np.float64)

    def _compute_localized_boxes(self, boxes):
        # Small interior views tolerate a few pixels of boundary uncertainty.
        # At least two views must agree; an isolated high-scoring patch is insufficient.
        views = []
        for x1, y1, x2, y2 in boxes:
            for inset in (0., .04, .08):
                dx, dy = int((x2-x1)*inset), int((y2-y1)*inset)
                views.append((x1+dx, y1+dy, x2-dx, y2-dy))
        crops = np.stack([crop_resize(self.frame, b, self.in_size) for b in views])
        feats = self.backbone(preprocess_batch(crops))
        raw = self.clf.cosine_batch(feats, self.data)
        color = self._hist_gate(crops, raw.copy(), white_sv=True)
        color_reasons = self._hist_reasons.copy()
        colour_compensated=set()
        for bi,box in enumerate(boxes):
            part=slice(bi*3,bi*3+3)
            for ci in range(len(self.data.labels)):
                if not self._generic_class(ci) or np.count_nonzero(color[part,ci]>=0)>=2 \
                        or np.count_nonzero(raw[part,ci]>=.85)<2:continue
                if self._registered_appearance_similarity(box,ci)<.80:continue
                pattern=(self._pattern_cache.get((ci,tuple(box)))
                    if getattr(self,'_pattern_cache_key',None)==(self.frame_idx,self._search_epoch) else None)
                if pattern is None and self._foreground_generic_identity() \
                        and self._generic_identity_hint_bank.get(ci):
                    pattern=self._registered_pattern(box,ci)
                proof={} if pattern is None else pattern.get('geometry',{})
                spatial={} if pattern is None else pattern.get('spatial',{})
                dense={} if pattern is None else pattern.get('dense',{})
                strong=(proof.get('verified') and proof.get('inliers',0)>=16
                    and proof.get('ratio',0.)>=.65 and proof.get('extent_iou',0.)>=.80
                    and min(proof.get('source_coverage',[0.,0.]))>=.35)
                strong=strong or (spatial.get('verified') and spatial.get('inliers',0)>=12
                    and spatial.get('extent',0.)>=.80 and spatial.get('mean_ncc',0.)>=.90
                    and min(spatial.get('coverage',[0.,0.]))>=.35)
                strong=strong or (dense.get('verified') and dense.get('correlation',0.)>=.90
                    and dense.get('aligned_hog',0.)>=.90 and dense.get('extent_iou',0.)>=.80
                    and dense.get('valid_fraction',0.)>=.90 and dense.get('supported_tiles',0)>=6)
                if not strong:continue
                corrected=self._hist_gate(crops[part],raw[part].copy(),white_sv=True,
                    generic_compensation=(ci,))
                color[part,ci]=corrected[:,ci]
                color_reasons[part,ci]=self._hist_reasons[:,ci]
                if np.count_nonzero(corrected[:,ci]>=0)>=2:colour_compensated.add((bi,ci))
        background = self._bg_gate(feats, color.copy())
        negative = self._neg_gate(feats, background.copy())
        negative_without_background = self._neg_gate(feats, color.copy()).reshape(len(boxes),3,-1)
        shaped = [v.reshape(len(boxes), 3, -1)
                  for v in (raw, color, background, negative)]
        output = np.full((len(boxes), len(self.data.labels)), -1., dtype=np.float64)
        whites=self._bright_localizer_flags()
        self._last_localized_evidence = []
        for bi, box in enumerate(boxes):
            records = {}
            # Exclude the silhouette: a wall's borders are not object texture.
            x1, y1, x2, y2 = box
            dx, dy = int((x2-x1)*.08), int((y2-y1)*.08)
            internal = self._shape_desc((x1+dx, y1+dy, x2-dx, y2-dy))
            for ci in range(len(self.data.labels)):
                rows = [v[bi, :, ci] for v in shaped]
                valid = rows[-1][np.isfinite(rows[-1]) & (rows[-1] >= 0)]
                if len(valid) >= 2:
                    # Allow three points for boundary uncertainty, but prevent
                    # a high silhouette from lifting a weak interior over it.
                    pair = np.sort(valid)[-2:]
                    output[bi, ci] = float(min(pair.mean(),pair[0]+.03))
                reason = "通过"
                if len(valid) < 2:
                    if np.count_nonzero(rows[1] >= 0) < 2:
                        failures = color_reasons[bi*3:(bi+1)*3, ci]
                        reason = "颜色门:" + next((str(r) for r in failures if r != "通过"), "视图不足")
                    elif np.count_nonzero(rows[2] >= 0) < 2:
                        reason = "背景门:与注册背景过近"
                    else:
                        reason = "负样本门:与负样本过近"
                ref = self.data.class_shape(ci)
                flat = ref is not None and float(ref[0])>.05 \
                       and float(internal[0])<max(.008,.2*float(ref[0]))
                if flat:
                    output[bi,ci] = -1.
                    if reason == "通过":
                        reason = "内部结构过少，疑似平坦背景"
                refinement=None
                if whites[ci] and reason=="通过" and not flat:
                    output[bi,ci],rows,refinement=self._refine_borderline_localized_views(
                        box,ci,float(output[bi,ci]),rows)
                    valid=rows[-1][np.isfinite(rows[-1]) & (rows[-1]>=0)]
                aligned=None
                # A transparent full crop can differ from the close registration
                # view even while the learned structure is visible. Add CURRENT
                # aligned views, with all original gates and the same lock line.
                # These are feature scores, not an average with NCC/HOG scores.
                if whites[ci] and not flat and output[bi,ci]<self.args.scan_cos_th+.02:
                    pattern=self._registered_pattern(box,ci)
                    hog=self._registered_appearance_similarity(box,ci)
                    strong=pattern is not None and self._registered_identity_ok(hog,pattern,ci)
                    if strong:
                        px,py,pw,ph=pattern['patch'];w,h=x2-x1,y2-y1
                        aligned_boxes=[]
                        for pad in (0.,.12):
                            aligned_boxes.append((max(x1,int(x1+(px-pad*pw)*w/96)),
                                max(y1,int(y1+(py-pad*ph)*h/96)),
                                min(x2,int(x1+(px+pw+pad*pw)*w/96)),
                                min(y2,int(y1+(py+ph+pad*ph)*h/96))))
                        aligned_crops=np.stack([crop_resize(self.frame,v,self.in_size) for v in aligned_boxes])
                        aligned_features=self.backbone(preprocess_batch(aligned_crops))
                        ar=self.clf.cosine_batch(aligned_features,self.data)
                        ac=self._hist_gate(aligned_crops,ar.copy(),white_sv=True)
                        ab=self._bg_gate(aligned_features,ac.copy())
                        an=self._neg_gate(aligned_features,ab.copy())
                        combined=[np.concatenate((old,new[:,ci]))for old,new in zip(rows,(ar,ac,ab,an))]
                        good=combined[-1][np.isfinite(combined[-1])&(combined[-1]>=self.args.scan_cos_th+.02)]
                        if len(good)>=2:
                            pair=np.sort(good)[-2:]
                            output[bi,ci]=float(min(pair.mean(),pair[0]+.03))
                            rows=combined;valid=rows[-1][np.isfinite(rows[-1])&(rows[-1]>=0)]
                            reason='通过：注册结构对齐视图复核'
                            aligned=dict(pattern=round(pattern['similarity'],3),boxes=aligned_boxes)
                colour_aligned=None
                if not whites[ci] and not flat and output[bi,ci]<self.args.scan_cos_th+.02 \
                        and np.count_nonzero(rows[1]>=0)>=2 \
                        and np.count_nonzero(negative_without_background[bi,:,ci]>=0)>=2 \
                        and (self._generic_class(ci) or chromatic_model(self.data.class_hist(ci)) is not None):
                    reference=self._registered_appearance_similarity(box,ci)
                    pattern=self._registered_pattern(box,ci) if reference>=.65 else None
                    background_clear=np.count_nonzero(rows[2]>=0)>=2
                    geometric=self._geometry_identity_ok(pattern,ci)
                    generic_layout=self._generic_class(ci) and self._registered_identity_ok(reference,pattern,ci)
                    if pattern is not None and (geometric or pattern['similarity']>=(.68 if background_clear else .78)) and reference>=.65 \
                            and (self._colour_boundary(box,ci) is not None
                                 or (generic_layout and self._localized_geometry_ok(box,ci))):
                        # The learned registration may be the title, rather
                        # than the whole opaque cover. Classify aligned
                        # CURRENT title views. Full-cover colour and negatives
                        # remain mandatory. A background prototype made from
                        # the rest of this SAME cover needs stronger spatial
                        # identity and independent core background checks.
                        px,py,pw,ph=pattern['patch'];w,h=x2-x1,y2-y1
                        ax1,ay1=max(x1,int(x1+px*w/96)),max(y1,int(y1+py*h/96))
                        ax2,ay2=min(x2,int(x1+(px+pw)*w/96)),min(y2,int(y1+(py+ph)*h/96))
                        aligned_boxes=[]
                        for inset in (0.,.04,.08):
                            dx,dy=int((ax2-ax1)*inset),int((ay2-ay1)*inset)
                            aligned_boxes.append((ax1+dx,ay1+dy,ax2-dx,ay2-dy))
                        if pw*ph>=.15*96*96 and all(min(b[2]-b[0],b[3]-b[1])>=16 for b in aligned_boxes):
                            quads=[]
                            if pattern.get('quad') is not None:
                                quad=np.float32(pattern['quad'])*np.float32([w/96.,h/96.])+np.float32([x1,y1])
                                centre=quad.mean(axis=0)
                                quads=[centre+(quad-centre)*(1.-2.*inset) for inset in (0.,.04,.08)]
                                iw,ih=self.in_size;destination=np.float32([[0,0],[iw-1,0],[iw-1,ih-1],[0,ih-1]])
                                acrops=np.stack([illum_norm(cv2.warpPerspective(self.frame,
                                    cv2.getPerspectiveTransform(q.astype(np.float32),destination),self.in_size)) for q in quads])
                            else:
                                acrops=np.stack([crop_resize(self.frame,b,self.in_size) for b in aligned_boxes])
                            afeats=self.backbone(preprocess_batch(acrops))
                            ar=self.clf.cosine_batch(afeats,self.data)
                            ac=self._hist_gate(acrops,ar.copy())
                            ab=self._bg_gate(afeats,ac.copy());an=self._neg_gate(afeats,ab.copy())
                            colour_aligned=dict(boxes=aligned_boxes,pattern=round(pattern['similarity'],3),
                                raw=ar[:,ci].round(3).tolist(),color=ac[:,ci].round(3).tolist(),
                                color_reasons=self._hist_reasons[:,ci].tolist(),
                                background=ab[:,ci].round(3).tolist(),negative=an[:,ci].round(3).tolist())
                            colour_aligned['background_conflict']=not background_clear
                            if quads:colour_aligned['quads']=[q.round(2).tolist() for q in quads]
                            good=an[:,ci][an[:,ci]>=self.args.scan_cos_th+.04]
                            if len(good)>=2:
                                pair=np.sort(good)[-2:];output[bi,ci]=float(min(pair.mean(),pair[0]+.03))
                                reason='通过：完整彩色边界与学习文字视图核验'
                records[ci] = dict(box=tuple(int(v) for v in box),
                                   colour_compensated=(bi,ci) in colour_compensated,
                                   frame=self.frame_idx,
                                   capture_time=getattr(self, '_capture_time', None),
                                   raw=rows[0].round(3).tolist(),
                                   color=rows[1].round(3).tolist(),
                                   color_reasons=color_reasons[bi*3:(bi+1)*3,ci].tolist(),
                                   background=rows[2].round(3).tolist(),
                                   negative=rows[3].round(3).tolist(),
                                   valid_views=int(len(valid)), score=round(float(output[bi, ci]),3),
                                   lock_views=int(np.count_nonzero(rows[-1]>=self.args.scan_cos_th+.02)),
                                   internal_structure=np.round(internal,3).tolist(),
                                   reason=reason)
                if refinement is not None:
                    records[ci]["boundary_view_refinement"]=refinement
                if aligned is not None:
                    records[ci]['registered_aligned_views']=aligned
                if colour_aligned is not None:
                    records[ci]['colour_registered_views']=colour_aligned
                direct = negative_without_background[bi,:,ci]
                direct = direct[np.isfinite(direct) & (direct>=0)]
                pair = np.sort(direct)[-2:]
                records[ci]["without_background_score"] = (
                    float(min(pair.mean(),pair[0]+.03)) if len(direct)>=2 and not flat else -1.)
            self._last_localized_evidence.append(records)
        return output

    def _init_target_motion(self, tgt):
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        tgt["motion_gray"] = gray
        tgt["motion_points"] = (self._seed_generic_motion(tgt['box'],tgt['cls'],gray)
                                if self._generic_class(tgt['cls']) else
                                seed_points(self.frame,gray,tgt['box']))
        tgt["motion_ref_age"] = 0
        tgt['motion_capture_time'] = getattr(self, '_capture_time', None)
        x1,y1,x2,y2 = tgt["box"]
        tgt["motion_corners"] = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]],np.float64)
        if self._generic_class(tgt['cls']):
            tgt['motion_box']=tuple(tgt['box'])
        tgt.setdefault("velocity", (0., 0.))

    def _localized_geometry_ok(self, box, ci):
        w,h = box[2]-box[0], box[3]-box[1]
        if min(w,h) < 16:
            return False
        ref = self.data.class_ar(ci)
        learned = max(ref, 1./ref)
        observed = max(w/h, h/w)
        ratio=observed/learned
        if .65 <= ratio <= 1.6:
            return True
        # Perspective and segmentation padding can shorten a genuine object's
        # observed aspect. Extend only a narrow border range with strong
        # CURRENT registered structure, never for an arbitrary bright blob.
        if .55 <= ratio <= 1.8:
            pattern=self._registered_pattern(box,ci)
            return bool(pattern is not None and pattern['similarity']>=.82
                        and self._registered_appearance_similarity(box,ci)>=.52)
        return False

    def _bright_seed_has_structure(self, box, ci):
        ref = self.data.class_shape(ci)
        if ref is None or float(ref[0])<=.05:
            return True
        x1,y1,x2,y2 = box
        dx,dy = int((x2-x1)*.08),int((y2-y1)*.08)
        return float(self._shape_desc((x1+dx,y1+dy,x2-dx,y2-dy))[0])>=.15*float(ref[0])

    def _registered_appearance_similarity(self, box, ci):
        # Candidate sources and admission gates often ask about the same crop.
        # Cache only within this frame/learning epoch; never reuse old identity.
        key=(self.frame_idx,self._search_epoch)
        if getattr(self,'_appearance_cache_key',None)!=key:
            self._appearance_cache_key=key
            self._appearance_descriptors={}
            self._appearance_scores={}
        score_key=(ci,tuple(box))
        if score_key in self._appearance_scores:
            return self._appearance_scores[score_key]
        if tuple(box) not in self._appearance_descriptors:
            self._appearance_descriptors[tuple(box)]=describe_appearance(self.frame,box)
        descriptor=self._appearance_descriptors[tuple(box)]
        if descriptor is None:
            return 0.
        score=max((float(descriptor@r) for r in self.data.class_visual_refs(ci)),default=0.)
        self._appearance_scores[score_key]=score
        return score

    def _foreground_generic_identity(self):
        return self._async_search is not None and not getattr(self,'_in_search_worker',False)

    @staticmethod
    def _spatial_hint_key(reference):
        # Fine references retain the global learned view/pose metadata even
        # when transported from a worker snapshot. Object ids do not survive
        # such copies. Older selected 24px references use local view indices,
        # so retain object identity for those rather than merging poses.
        if reference.get('patch_size')==16:
            return ('spatial',16,reference['reference'],reference['flip'],reference['rotation'])
        return ('spatial',id(reference))

    def _remember_generic_hint(self,ci,pattern):
        """Keep bounded LEARNED reference choices, never past acceptance scores."""
        if pattern is None:return
        dense=pattern.get('dense',{})
        template=pattern.get('_hint_pattern')
        # Bare geometry is not a reusable learned reference. Selected spatial
        # and ORB/SIFT references are retained only with their learned data.
        spatial=pattern.get('_hint_spatial_ref');scene=pattern.get('_hint_scene_ref')
        if pattern.get('_hint_dense_ref') is not None:
            signature=('seed',id(pattern['_hint_dense_ref']))
        elif dense:
            signature=('dense',dense.get('reference'),dense.get('rotation'),dense.get('flip'))
        elif spatial is not None:
            signature=(self._spatial_hint_key(spatial) if spatial.get('patch_size')==16
                       else ('spatial',id(spatial['gray'])))
        elif scene is not None:
            signature=('scene',pattern.get('_hint_scene_method'),id(scene[1]))
        elif template is not None:
            signature=('ncc',id(template),tuple(pattern.get('_hint_size',())))
        else:return
        banks=getattr(self,'_generic_identity_hint_bank',{})
        bank=[(key,value) for key,value in banks.get(ci,[]) if key!=signature]
        banks[ci]=[(signature,pattern)]+bank[:2]
        self._generic_identity_hint_bank=banks
        self._generic_identity_hints[ci]=pattern

    @contextmanager
    def _generic_reference_trial(self,ci,protect_choice=False):
        """Keep speculative learned choices private until final acceptance."""
        if not self._generic_class(ci):
            yield lambda:None
            return
        saved={}
        for name in ('_generic_identity_hint_bank','_generic_identity_hints'):
            mapping=getattr(self,name)
            saved[name]=(ci in mapping,mapping.get(ci))
        state=self._identity_task_states.get(ci,{})
        choices={name:(name in state,state.get(name)) for name in ('next','retry_incomplete')}
        committed=False
        def commit():
            nonlocal committed
            committed=True
        try:
            yield commit
        finally:
            if not committed:
                for name,(present,value) in saved.items():
                    mapping=getattr(self,name)
                    if present:mapping[ci]=value
                    else:mapping.pop(ci,None)
                state=self._identity_task_states.get(ci,{})
                # Keep incomplete primary measurements advancing. A rejected
                # candidate must not replace an existing target's next choice;
                # nor may an identity-only success rejected by later gates.
                if protect_choice or state.get('status')=='verified':
                    for name,(present,value) in choices.items():
                        if present:state[name]=value
                        else:state.pop(name,None)

    def _recheck_scene_hint(self,crop,hint,deadline):
        reference=hint.get('_hint_scene_ref')
        if reference is None:return None
        if time.perf_counter()>=deadline:return IDENTITY_INCOMPLETE
        method=hint.get('_hint_scene_method','orb')
        progress={}
        candidates=project_reference_candidates(crop,[reference],method,
            time_budget=max(0.,deadline-time.perf_counter()),progress=progress)
        fits=[proof for _,proof in candidates if proof.get('extent_iou',0.)>=.75]
        if not fits:return None if progress['completed'] else IDENTITY_INCOMPLETE
        proof=max(fits,key=lambda p:p['inliers'])
        h,w=crop.shape[:2];quad=np.float32(proof['full_quad'])*[96./w,96./h]
        lo,hi=quad.min(axis=0),quad.max(axis=0)
        return dict(similarity=-1.,geometry=proof,full_quad=quad.tolist(),patch=tuple(lo)+tuple(hi-lo),
                    _hint_scene_ref=reference,_hint_scene_method=method)

    def _current_identity_tasks(self,box,ci,deadline):
        """Continue reference selection across frames; never accumulate scores."""
        prepared=time.perf_counter()
        if prepared>=deadline:
            self._identity_task_states.setdefault(ci,{}).update(
                status='pending',frame=self.frame_idx,box=tuple(box),checked=0,incomplete='preparation',
                events=[],completed=[],last_task=None,reference_similarity=None)
            return None
        x,y,r,b=map(int,box);crop=self.frame[y:b,x:r]
        hints=[v for _,v in self._generic_identity_hint_bank.get(ci,[])]
        reference=self._registered_appearance_similarity(box,ci)
        tasks=[]
        for hint in hints:
            if hint.get('_hint_point_ref') is not None:
                tasks.append((('points',id(hint['_hint_point_ref'])),
                    lambda h=hint:recheck_learned_points(crop,h,deadline)))
            if hint.get('_hint_colour_ref') is not None:
                tasks.append((('colour',id(hint['_hint_colour_ref'][1])),
                    lambda h=hint:self._check_selected_colour(crop,h)))
            if hint.get('_hint_dense_seed') is not None and hint.get('_hint_dense_ref') is not None:
                ref=hint['_hint_dense_ref']
                tasks.append((('seed',id(ref)),lambda h=hint:self._check_dense_seed(crop,h)))
            if hint.get('dense') and (hint.get('_hint_dense_seed') is None or hint.get('_hint_dense_ref') is None):
                proof=hint['dense'];key=('dense',proof.get('reference'),proof.get('rotation'),proof.get('flip'))
                tasks.append((key,lambda h=hint:recheck_dense_hint(crop,self.data.class_dense_refs(ci),h)))
            if hint.get('_hint_pattern') is not None:
                for delta in (0.,-4.,4.):
                    tasks.append((('ncc',id(hint['_hint_pattern']),delta),
                        lambda h=hint,a=delta:recheck_registration_hint(crop,h,angle_delta=a)))
            if hint.get('_hint_spatial_ref') is not None:
                tasks.append((self._spatial_hint_key(hint['_hint_spatial_ref']),
                    lambda h=hint:recheck_spatial_hint(crop,h,deadline=deadline)))
            scene_ref=hint.get('_hint_scene_ref')
            if scene_ref is not None:
                tasks.append((('scene',hint.get('_hint_scene_method','orb'),id(scene_ref[1])),
                    lambda h=hint:self._recheck_scene_hint(crop,h,deadline)))
        tasks.extend([(('direct',),lambda:match_dense_reference(crop,self.data.class_dense_refs(ci),
            max_attempts=2,max_iterations=0,prefer_direct=True))])
        # Keep all existing checks first. If none succeeds, NCC must still be
        # available even when acquisition selected a different proof method.
        # Re-select LEARNED templates on CURRENT pixels, with unchanged gates.
        tasks.append((('indexed_ncc',),lambda:self._check_indexed_registration(crop,ci)))
        # Different hints can refer to the same learned data. One current
        # measurement per reference/method is sufficient within a round.
        tasks=list(dict(tasks).items())
        if any((h.get('_hint_spatial_ref') or {}).get('patch_size')==16 for h in hints):
            # A changed view can reject the selected learning image while
            # another saved view still proves the CURRENT crop. Try learned
            # reference reselection before expensive unrelated fallbacks.
            spatial_tasks=[task for task in tasks if task[0][0]=='spatial']
            # One task per learned choice can advance across frames under a
            # small per-class budget. Never restart the whole multi-view
            # packet when only its first references fit before the deadline.
            selected=set()
            for hint in hints:
                ref=hint.get('_hint_spatial_ref')
                if ref is not None and ref.get('patch_size')==16:
                    selected.add((ref['reference'],ref['flip'],ref['rotation']))
            choices=spatial_reference_choices(crop,self.data.class_spatial_refs(ci,patch_size=16))
            reselection=[(('spatial_reselect',)+self._spatial_hint_key(ref)[1:],lambda ref=ref:recheck_spatial_hint(
                crop,dict(_hint_spatial_ref=ref),deadline=deadline)) for ref in choices
                if (ref['reference'],ref['flip'],ref['rotation']) not in selected]
            tasks=spatial_tasks+reselection+[task for task in tasks if task[0][0]!='spatial']
        perf=getattr(self,'_generic_perf',{})
        perf['identity_prepare_ms']=perf.get('identity_prepare_ms',0.)+(time.perf_counter()-prepared)*1000.
        self._generic_perf=perf
        states=getattr(self,'_identity_task_states',{});state=states.setdefault(ci,{})
        keys=[k for k,_ in tasks];start=keys.index(state['next']) if state.get('next') in keys else 0
        completed=set(state.get('completed',())) & set(keys)
        # Negative measurements belong to exact current pixels. Only the
        # next task selection may cross frames, not completed failures from a
        # different image or a different crop in the same image.
        if (state.get('status') in ('verified','rejected')
                or state.get('frame')!=self.frame_idx
                or tuple(state.get('box',()))!=tuple(box)):
            completed=set()
        state.update(frame=self.frame_idx,box=tuple(box),checked=0,incomplete=None,
                     reference_similarity=round(float(reference),4),events=[])
        state['status']='pending'
        for step in range(len(tasks)):
            index=(start+step)%len(tasks);key,check=tasks[index]
            if key in completed:continue
            if time.perf_counter()>=deadline:
                state.update(next=key,incomplete=key[0])
                state['events'].append(identity_attempt(key[0],None,reference,'not_started',0.,
                    (deadline-time.perf_counter())*1000.))
                break
            check_start=time.perf_counter()
            pattern=check()
            state.update(last_task=key[0],checked=state['checked']+1)
            perf=getattr(self,'_generic_perf',{})
            metric='identity_'+key[0]+'_ms'
            elapsed=(time.perf_counter()-check_start)*1000.
            perf[metric]=perf.get(metric,0.)+elapsed
            self._generic_perf=perf
            if pattern is IDENTITY_INCOMPLETE:
                state['events'].append(identity_attempt(key[0],None,reference,'incomplete',elapsed,
                    (deadline-time.perf_counter())*1000.))
                # Give a single-reference task one fresh full-budget retry,
                # then rotate if it still cannot finish. Only the task choice
                # persists; no camera pixels or successful evidence carry over.
                single_fine=(key[0]=='spatial' and len(key)==5 and key[1]==16)
                retry=(key[0]=='spatial_reselect' or single_fine) and state.get('retry_incomplete')!=key
                state.update(next=key if retry else tasks[(index+1)%len(tasks)][0],incomplete=key[0],
                             retry_incomplete=key if retry else None)
                break
            state.pop('retry_incomplete',None)
            identity_ok=self._registered_identity_ok(reference,pattern,ci)
            state['events'].append(identity_attempt(key[0],pattern,reference,
                'verified' if identity_ok else 'no_match' if pattern is None else 'identity_rejected',
                elapsed,(deadline-time.perf_counter())*1000.))
            if identity_ok:
                next_key=(('ncc',id(pattern['_hint_pattern']),0.)
                    if key==('indexed_ncc',) and pattern.get('_hint_pattern') is not None else key)
                if key[0]=='spatial_reselect' and pattern.get('_hint_spatial_ref') is not None:
                    next_key=self._spatial_hint_key(pattern['_hint_spatial_ref'])
                state.update(status='verified',next=next_key,completed=[])
                self._identity_task_states=states
                return pattern
            completed.add(key)
            state['next']=tasks[(index+1)%len(tasks)][0]
        else:state['status']='rejected'
        # Only task selection crosses frames. No score or successful evidence
        # accumulates; any acceptance must come from this frame's check.
        state['completed']=list(completed) if state['status']=='pending' else []
        self._identity_task_states=states
        return None

    def _check_indexed_registration(self,crop,ci):
        patterns=indexed_registration_patterns(crop,self.data.class_dense_refs(ci),
            self.data.class_reference_views(ci),self.data._generic_pattern_groups.setdefault(ci,{}),groups=3)
        return match_colour_registration_pattern(crop,patterns,(),min_extent=2./3.,rotation_limit=2)

    def _current_geometry_pattern(self,crop,ci):
        geometry=match_colour_geometry(crop,self.data.class_colour_geometry_refs(ci),min_extent=.75)
        return None if geometry is None else dict(geometry,similarity=-1.)

    def _check_selected_colour(self,crop,hint):
        result=match_colour_geometry(crop,[hint['_hint_colour_ref']],min_extent=.75,max_iterations=500)
        if result is None:return None
        metadata={k:v for k,v in hint.items() if k.startswith('_hint_')}
        if result.get('full_quad') is not None and hint.get('_hint_dense_ref') is not None:
            flipped,rotation=hint.get('_hint_reference_pose',(False,0))
            order=np.array([1,0,3,2] if flipped else [0,1,2,3])
            oriented=np.float32(result['full_quad'])[np.roll(order,-int(rotation))]
            metadata['_hint_dense_seed']=dict(hint.get('_hint_dense_seed',{}),full_quad=oriented.tolist())
        elif hint.get('_hint_dense_ref') is None:
            metadata.pop('_hint_dense_seed',None)
        return dict(metadata,similarity=-1.,**result)

    def _check_dense_seed(self,crop,hint):
        # A geometric/NCC hint need not contain a dense learned reference.
        # Incomplete hints cannot authorize this check or terminate the GUI.
        if hint.get('_hint_dense_ref') is None or hint.get('_hint_dense_seed') is None:return None
        result=recheck_dense_hint(crop,[hint['_hint_dense_ref']],hint['_hint_dense_seed'])
        if result is None:return None
        metadata={k:v for k,v in hint.items() if k.startswith('_hint_')}
        metadata['_hint_dense_seed']=dict(dense=result['dense'],full_quad=result['full_quad'])
        return dict(metadata,**result)

    def _identity_pending(self,ci):
        return getattr(self,'_identity_task_states',{}).get(ci,{}).get('status')=='pending'

    def _registered_pattern(self, box, ci):
        if not (self._generic_class(ci) and self._foreground_generic_identity()):
            return self._registered_pattern_impl(box,ci)
        started=time.perf_counter()
        try:
            return self._registered_pattern_impl(box,ci)
        finally:
            perf=getattr(self,'_generic_perf',{})
            perf['generic_identity_ms']=perf.get('generic_identity_ms',0.)+(time.perf_counter()-started)*1000.
            perf['generic_identity_calls']=perf.get('generic_identity_calls',0)+1
            self._generic_perf=perf

    def _registered_pattern_impl(self, box, ci):
        foreground=self._generic_class(ci) and self._foreground_generic_identity()
        fast=foreground or getattr(self,'_generic_identity_fast',False)
        key = (self.frame_idx, self._search_epoch)
        if getattr(self, '_pattern_cache_key', None) != key:
            self._pattern_cache_key = key
            self._pattern_cache = {}
        if getattr(self,'_pattern_fast_miss_key',None)!=key:
            self._pattern_fast_miss_key=key
            self._pattern_fast_misses=set()
        cache_key = (ci, tuple(box))
        cached=self._pattern_cache.get(cache_key)
        if cached is not None:self._pattern_fast_misses.discard(cache_key)
        # A completed quick failure is not a completed full failure. Keep it
        # reusable by quick callers, but allow an explicit full-mode upgrade.
        if not fast and cached is None and cache_key in self._pattern_fast_misses:
            self._pattern_cache.pop(cache_key,None)
            self._pattern_fast_misses.discard(cache_key)
        if cached is not None and cached.get('_fast_only') and not fast:
            del self._pattern_cache[cache_key]
        if cache_key not in self._pattern_cache:
            x1, y1, x2, y2 = map(int, box)
            if foreground:
                budget_key=(self.frame_idx,self._search_epoch)
                if getattr(self,'_identity_budget_key',None)!=budget_key:
                    self._identity_budget_key=budget_key;self._identity_deadlines={}
                    self._identity_time_used=0.
                # Absent learned classes do not consume a foreground slice.
                # Share with current targets and this incoming identity, while
                # retaining the frame's total soft matching budget.
                active={t['cls'] for t in self.scan_targets if self._generic_class(t['cls'])}
                active.add(ci)
                now=time.perf_counter()
                # Finished identities donate their unused matching time.
                # Reserve equal shares of the REMAINING frame budget for
                # identities not yet served; retain each class's fixed deadline
                # once assigned, so repeated crops cannot extend its slice.
                remaining=max(0.,.012-self._identity_time_used)
                unserved=active-set(self._identity_deadlines)
                share=remaining/max(1,len(unserved))
                deadline=min(self._identity_deadlines.setdefault(ci,now+share),now+remaining)
                if time.perf_counter()>=deadline:
                    # No check ran for this crop. Do not attach a pending
                    # status to a previous frame's box or successful events.
                    self._identity_task_states.setdefault(ci,{}).update(
                        status='pending',frame=self.frame_idx,box=tuple(box),
                        checked=0,incomplete='frame_budget',events=[],completed=[],
                        last_task=None,reference_similarity=None)
                    self._pattern_cache[cache_key]=None
                    self._pattern_fast_misses.add(cache_key)
                    return None
                started=time.perf_counter()
                try:
                    pattern=self._current_identity_tasks(box,ci,deadline)
                finally:
                    self._identity_time_used+=time.perf_counter()-started
                self._pattern_cache[cache_key]=pattern
                if pattern is None:self._pattern_fast_misses.add(cache_key)
                return pattern
            if self._generic_class(ci):
                dense=match_dense_reference(self.frame[y1:y2,x1:x2],self.data.class_dense_refs(ci),
                    max_iterations=24,
                    prefer_direct=True)
                if dense is not None:
                    self._pattern_cache[cache_key]=dense
                    return dense
            if self._bright_localizer_flags()[ci]:
                self._pattern_cache[cache_key]=match_registration_pattern(
                    self.frame[y1:y2,x1:x2],self.data.class_pattern_refs(ci))
            else:
                if self._generic_class(ci) and fast:
                    patterns=indexed_registration_patterns(self.frame[y1:y2,x1:x2],
                        self.data.class_dense_refs(ci),self.data.class_reference_views(ci),
                        self.data._generic_pattern_groups.setdefault(ci,{}))
                else:
                    patterns=self.data.class_pattern_refs(ci)
                baseline=match_colour_registration_pattern(
                    self.frame[y1:y2,x1:x2],patterns,
                    self.data.class_colour_geometry_refs(ci),
                    rotation_limit=8)
                if not self._generic_class(ci) or self._registered_identity_ok(
                        self._registered_appearance_similarity(box,ci),baseline,ci):
                    self._pattern_cache[cache_key]=baseline
                    return baseline
                # Supplement a failed small-reference comparison with saved
                # full-resolution references.
                # The projected full extent must agree with this candidate;
                # matching a logo alone cannot authorize a large scene box.
                if self._generic_class(ci) and not fast:
                    candidates=self._project_generic_references(self.frame[y1:y2,x1:x2],ci,min_extent=.75)
                    fitting=[proof for _,proof in candidates if proof['extent_iou']>=.75]
                    if fitting:
                        proof=max(fitting,key=lambda p:p['inliers'])
                        quad=np.asarray(proof['full_quad'],np.float32)*[96./(x2-x1),96./(y2-y1)]
                        lower,upper=quad.min(axis=0),quad.max(axis=0)
                        self._pattern_cache[cache_key]=dict(similarity=-1.,geometry=proof,
                            full_quad=quad.tolist(),patch=tuple(lower)+tuple(upper-lower))
                        return self._pattern_cache[cache_key]
                # Supplement failed existing evidence only: keep stronger
                # geometry available to its original feature-recovery path.
                if self._generic_class(ci) and self._registered_appearance_similarity(box,ci)>=.70:
                    spatial=match_spatial_reference(self.frame[y1:y2,x1:x2],
                        self.data.class_spatial_refs(ci))
                    if spatial is not None:
                        self._pattern_cache[cache_key]=spatial
                        return spatial
                    fine=match_fine_spatial_reference(self.frame[y1:y2,x1:x2],
                        self.data.class_spatial_refs(ci,patch_size=16))
                    if fine is not None:
                        self._pattern_cache[cache_key]=fine
                        return fine
                # Preserve all existing successful paths, including geometric
                # and distributed matches. Only their failure gets the extra
                # full-extent NCC search with distributed interior texture.
                # NCC identity already requires HOG >= .70; avoid spending
                # the search budget on crops that cannot pass that gate.
                if self._registered_appearance_similarity(box,ci)>=.70:
                    complete=match_colour_registration_pattern(
                        self.frame[y1:y2,x1:x2],patterns,
                        self.data.class_colour_geometry_refs(ci),min_extent=2./3.,
                        rotation_limit=8)
                    if self._registered_identity_ok(
                            self._registered_appearance_similarity(box,ci),complete,ci):
                        self._pattern_cache[cache_key]=complete
                        return complete
                self._pattern_cache[cache_key]=(dict(baseline,_fast_only=True) if baseline is not None
                    and self._generic_class(ci) and fast else baseline)
                if fast and self._generic_class(ci) and self._pattern_cache[cache_key] is None:
                    self._pattern_fast_misses.add(cache_key)
        return self._pattern_cache[cache_key]

    def _registered_body_shape(self, box, ci):
        references = self.data.class_body_shapes(ci)
        if not references:
            return None, 0., 1.
        x1, y1, x2, y2 = map(int, box)
        contour, fraction = bright_body_contour(self.frame[y1:y2, x1:x2])
        if contour is None:
            return float('inf'), fraction, float('inf')
        distance = min(float(cv2.matchShapes(contour, reference,
                        cv2.CONTOURS_MATCH_I1, 0.)) for reference in references)
        def elongation(c):
            _, (w, h), _ = cv2.minAreaRect(c)
            return max(w, h) / max(1., min(w, h))
        observed = elongation(contour)
        ratio = min(max(observed, elongation(ref)) / min(observed, elongation(ref))
                    for ref in references)
        return distance, fraction, ratio

    def _localized_candidate_order(self, boxes, scores, ci, threshold=None):
        """Compare alternative boundaries of the same foreground component."""
        threshold=self.args.scan_cos_th+.02 if threshold is None else threshold
        remaining=set(range(len(boxes)))
        references={}
        density_key=(self.frame_idx,self._search_epoch)
        if getattr(self,'_extension_cache_key',None)!=density_key:
            self._extension_cache_key=density_key
            self._extension_cache={}
        def density(box):
            if box not in self._extension_cache:
                self._extension_cache[box]=transparent_extension_density(self.frame,box)
            return self._extension_cache[box]
        ordered=[]
        while remaining:
            strongest=max(remaining,key=lambda j:float(scores[j]))
            pick=strongest
            alternatives=[j for j in remaining if float(scores[j])>=threshold
                          and _same_obj(boxes[j],boxes[strongest],.5)
                          and self._localized_geometry_ok(boxes[j],ci)
                          and self._shape_gate_ok(boxes[j],ci,True)]
            if len(alternatives)>1 and self.data.class_visual_refs(ci):
                for j in alternatives+[strongest]:
                    if j not in references:
                        references[j]=self._registered_appearance_similarity(boxes[j],ci)
                best=max(alternatives,key=lambda j:references[j])
                if references[best]>=.58 and references[best]>=references[strongest]+.015:
                    pick=best
                # A small internal wheel can match the same learned object.
                # Preserve a larger independently verified extent when it
                # contains that patch and keeps comparable registered support.
                reference=references.get(pick,0.)
                enclosing=[j for j in alternatives if j!=pick
                           and references[j]>=max(.60,reference-.04)
                           and _iou(boxes[j],boxes[pick])>=.05
                           and (max(0,min(boxes[j][2],boxes[pick][2])-max(boxes[j][0],boxes[pick][0]))
                                *max(0,min(boxes[j][3],boxes[pick][3])-max(boxes[j][1],boxes[pick][1])))
                               >=.85*(boxes[pick][2]-boxes[pick][0])*(boxes[pick][3]-boxes[pick][1])
                           and (boxes[j][2]-boxes[j][0])*(boxes[j][3]-boxes[j][1])
                               >1.3*(boxes[pick][2]-boxes[pick][0])*(boxes[pick][3]-boxes[pick][1])]
                if enclosing:
                    pick=max(enclosing,key=lambda j:(boxes[j][2]-boxes[j][0])*(boxes[j][3]-boxes[j][1]))
                # A real segmented full shell must not lose to an artificial
                # inset just because the inset resembles the close-up learned
                # thumbnail. Require stronger CURRENT core evidence, comparable
                # classification, visible endpoint support and actual measured
                # bounds; padding variants cannot authorize this expansion.
                anchor=boxes[pick]
                area=max(1,(anchor[2]-anchor[0])*(anchor[3]-anchor[1]))
                measured=[]
                for j in alternatives:
                    b=boxes[j]
                    if j==pick or b not in getattr(self,'_measured_boundary_boxes',set()):continue
                    ratio=((b[2]-b[0])*(b[3]-b[1]))/area
                    if not 1.20<=ratio<=1.75 or references[j]<.52 \
                            or float(scores[j])<float(scores[pick])-.03:continue
                    intersection=max(0,min(b[2],anchor[2])-max(b[0],anchor[0])) \
                        *max(0,min(b[3],anchor[3])-max(b[1],anchor[1]))
                    if intersection<.9*area or not endpoint_boundary_support(self.frame,b)[1]:continue
                    pattern=self._registered_pattern(b,ci)
                    if pattern is not None and pattern['similarity']>=.85:measured.append(j)
                if measured:
                    pick=max(measured,key=lambda j:(boxes[j][2]-boxes[j][0])*(boxes[j][3]-boxes[j][1]))
                # Internal markings establish identity; a separately measured
                # shell establishes extent. Resizing the full transparent shell
                # can reduce HOG similarity, so it need not beat its inner crop.
                anchor=boxes[pick]
                complete=[j for j in alternatives if j!=pick
                          and references[j]>=.55
                          and shell_extension_support(self.frame,anchor,boxes[j],self.data.class_ar(ci))>0.
                          and self._new_white_identity_ok(boxes[j],ci,float(scores[j]))]
                if complete:
                    pick=max(complete,key=lambda j:(boxes[j][2]-boxes[j][0])*(boxes[j][3]-boxes[j][1]))
                # Competing left/right tails share the same opaque body and
                # registered internal pattern. Prefer the direction with actual
                # neutral structure, rather than a higher-scoring face/window.
                for j in alternatives:
                    if j == pick or float(scores[j]) < float(scores[pick])-.08:
                        continue
                    a, b = boxes[pick], boxes[j]
                    height = min(a[3]-a[1], b[3]-b[1])
                    if height <= 0 or max(abs(a[1]-b[1]),abs(a[3]-b[3])) > .12*height:
                        continue
                    common_left, common_right = max(a[0],b[0]), min(a[2],b[2])
                    if common_right-common_left < .5*min(a[2]-a[0], b[2]-b[0]):
                        continue
                    if b[0]<a[0]-4 and b[2]<a[2]-4:
                        extra_b=(b[0],max(a[1],b[1]),common_left,min(a[3],b[3]))
                        extra_a=(common_right,max(a[1],b[1]),a[2],min(a[3],b[3]))
                    elif b[0]>a[0]+4 and b[2]>a[2]+4:
                        extra_b=(common_right,max(a[1],b[1]),b[2],min(a[3],b[3]))
                        extra_a=(a[0],max(a[1],b[1]),common_left,min(a[3],b[3]))
                    else:
                        continue
                    support_b=density(extra_b)
                    support_a=density(extra_a)
                    if support_b >= max(.025, 1.25*support_a+.01) \
                            and references.get(j,0.)>=references.get(pick,0.) \
                            and self._new_white_identity_ok(b,ci,float(scores[j])):
                        pick=j
                # Opposing clear-shell proposals can share the same learned
                # body while one extends into a dark shirt/face background.
                # Evaluate those disjoint sides before selecting a first
                # identity; full-crop scores dilute the wrong-side evidence.
                for j in alternatives:
                    if j==pick or float(scores[j])<float(scores[pick])-.08:
                        continue
                    side=competing_shell_evidence(self.frame,boxes[pick],boxes[j])
                    if side is not None and side['first_dark']>=.35 \
                            and side['second_dark']<=.15 and side['second_edges']>=.03 \
                            and references.get(j,0.)>=.52 \
                            and self._new_white_identity_ok(boxes[j],ci,float(scores[j])):
                        pick=j
            ordered.append(pick)
            remaining.remove(pick)
        return ordered

    def _trusted_white_colour_score(self, box, ci, evidence):
        """Resolve dark rectangular corners using a classified learned patch.

        Only the dark-pixel rejection may be bridged, only for a caller's
        confirmed track. The matched patch must independently pass color,
        classification, background and user negatives; full crops retain the
        background/negative checks and two lock-strength feature views.
        """
        reasons=evidence.get('color_reasons',[])
        if '暗像素偏多' not in reasons:
            return None
        pattern=self._registered_pattern(box,ci)
        if pattern is None or pattern['similarity']<.82:
            return None
        px,py,pw,ph=pattern['patch'];x1,y1,x2,y2=box;w,h=x2-x1,y2-y1
        patch=(max(x1,x1+int(px*w/96)),max(y1,y1+int(py*h/96)),
               min(x2,x1+int(np.ceil((px+pw)*w/96))),
               min(y2,y1+int(np.ceil((py+ph)*h/96))))
        crop=crop_resize(self.frame,patch,self.in_size)[None]
        features=self.backbone(preprocess_batch(crop))
        scores=self.clf.cosine_batch(features,self.data)
        scores=self._hist_gate(crop,scores,white_sv=True)
        scores=self._neg_gate(features,self._bg_gate(features,scores))
        lock=self.args.scan_cos_th+.02
        if float(scores[0,ci])<lock:
            return None
        views=[(x1+int(w*s),y1+int(h*s),x2-int(w*s),y2-int(h*s)) for s in (0.,.04,.08)]
        crops=np.stack([crop_resize(self.frame,v,self.in_size) for v in views])
        features=self.backbone(preprocess_batch(crops))
        raw=self.clf.cosine_batch(features,self.data)
        background=self._bg_gate(features,raw.copy())
        negative=self._neg_gate(features,background.copy())
        restored=np.array(evidence['negative'][:3],dtype=float)
        for j,reason in enumerate(reasons[:3]):
            if reason=='暗像素偏多':
                restored[j]=negative[j,ci]
        valid=restored[np.isfinite(restored)&(restored>=lock)]
        if len(valid)<2:
            return None
        pair=np.sort(valid)[-2:];score=float(min(pair.mean(),pair[0]+.03))
        evidence.update(score=round(score,3),reason='已确认物体：注册内部结构与颜色复核通过',
            registered_pattern_recovery=round(pattern['similarity'],3),colour_patch=patch,
            colour_patch_score=round(float(scores[0,ci]),3),restored_views=restored.round(3).tolist())
        return score

    def _registered_identity_ok(self,reference,pattern,ci):
        """Use complete learned reference geometry for new registrations.

        Historical cropped V2 references retain their compatibility policy;
        its fixed neutral-target placement must not constrain a new category.
        """
        if self._generic_class(ci) and pattern is None:
            return False
        if self._generic_class(ci) and pattern is not None:
            spatial=pattern.get('spatial',{})
            if spatial.get('verified'):
                # Correspondences must span the reference and project its
                # full boundary; a small matching logo is insufficient.
                return (reference>=.70 and spatial.get('inliers',0)>=(12 if spatial.get('patch_size')==16 else 8)
                        and spatial.get('extent',0.)>=.78
                        and min(spatial.get('coverage',[0.,0.]))>=.35
                        and spatial.get('mean_ncc',0.)>=.80)
            dense=pattern.get('dense',{})
            if dense.get('verified'):
                return (reference>=.5 and dense.get('extent_iou',0.)>=.78
                        and dense.get('correlation',0.)>=.82
                        and dense.get('aligned_hog',0.)>=.80
                        and dense.get('supported_tiles',0)>=4)
            proof=pattern.get('geometry',{})
            if proof.get('verified'):
                return proof.get('extent_iou',0.)>=.75 and reference>=.58
            patch=pattern.get('patch',(0,0,0,0))
            substantial=patch[2]*patch[3]>=.25*96*96
            # NCC templates represent the 56/64 registration interior. Infer
            # the WHOLE reference extent, including its rim, before admitting
            # identity. A local matching stripe/logo cannot stand for the ROI.
            x,y,w,h=map(float,patch)
            quad=np.asarray(pattern.get('quad',[[x,y],[x+w,y],[x+w,y+h],[x,y+h]]),np.float32)
            centre=quad.mean(axis=0);full=centre+(quad-centre)*(64./56.)
            lo,hi=full.min(axis=0),full.max(axis=0)
            intersection=float(np.prod(np.maximum(0,np.minimum(hi,96)-np.maximum(lo,0))))
            extent=intersection/max(1.,float(np.prod(hi-lo))+96*96-intersection)
            pattern['reference_extent_iou']=round(extent,4)
            return (extent>=2./3. and reference>=(.70 if substantial else .75)
                    and pattern['similarity']>=(.68 if substantial else .78))
        if pattern is not None and self._bright_localizer_flags()[ci]:
            patch=pattern.get('patch')
            if patch is not None:
                px,py,pw,ph=map(float,patch)
                cx=(px+.5*pw)/96.
                cy=(py+.5*ph)/96.
                if not (.40<=cx<=.82 and .30<=cy<=.75):
                    return False
        return registered_identity_supported(reference,pattern)

    def _geometry_identity_ok(self,pattern,ci):
        proof={} if pattern is None else pattern.get('geometry',{})
        return bool(proof.get('verified') and
                    (not self._generic_class(ci) or proof.get('extent_iou',0.)>=.75))

    def _generic_geometry_admission(self,box,ci,score):
        """Independent full-reference identity, with current feature vetoes.

        CNN similarity is not a probability. A distributed geometric match can
        identify a learned object despite weaker whole-crop features; it must
        not alter that feature score or override colour/background/negatives.
        """
        if not self._generic_class(ci) or score<0.:
            return False
        pattern=self._registered_pattern(box,ci)
        proof={} if pattern is None else pattern.get('geometry',{})
        dense={} if pattern is None else pattern.get('dense',{})
        strong_dense=bool(dense.get('verified') and dense.get('correlation',0.)>=.90
            and dense.get('aligned_hog',0.)>=.90 and dense.get('extent_iou',0.)>=.80
            and dense.get('valid_fraction',0.)>=.90 and dense.get('supported_tiles',0)>=6)
        strong_points=bool(proof.get('verified') and proof.get('extent_iou',0.)>=.80
                and proof.get('inliers',0)>=16 and proof.get('ratio',0.)>=.65
                and min(proof.get('source_coverage',[0.,0.]))>=.35
                and self._registered_appearance_similarity(box,ci)>=.58)
        if not (strong_dense or strong_points):
            return False
        self._score_localized_boxes([box])
        evidence=self._last_localized_evidence[0][ci]
        return all(sum(float(v)>=0 for v in evidence.get(name,[])[:3])>=2
                   for name in ('color','background','negative'))

    def _new_white_identity_ok(self,box,ci,score):
        # A high backbone similarity alone also occurs on folds and clothing.
        # Registered views provide independent support for every new identity.
        if not self.data.class_visual_refs(ci):
            return not self._generic_class(ci)
        reference=self._registered_appearance_similarity(box,ci);pattern=self._registered_pattern(box,ci)
        return self._registered_identity_ok(reference,pattern,ci) or bool(not self._bright_localizer_flags()[ci]
            and self._geometry_identity_ok(pattern,ci)
            and reference>=.65 and score>=self.args.scan_cos_th+.04)

    def _localized_boundary_consistent(self,box,target,score):
        old=target['box']
        if (self._generic_class(target['cls']) and not target.get('verified')
                and target.get('generic_hidden_motion')
                and time.perf_counter()<=target.get('generic_recovery_until',0.)
                and time.perf_counter()-target.get('generic_hidden_motion_time',0.)<=.2):
            old=target.get('motion_box',old)
        elif (self._generic_class(target['cls']) and not target.get('verified')
                and time.perf_counter()<=target.get('generic_recovery_until',0.)
                and time.perf_counter()-target.get('generic_recovery_box_time',0.)<=.15):
            old=target.get('generic_recovery_box',old)
        old_area=max(1,(old[2]-old[0])*(old[3]-old[1]))
        area=max(1,(box[2]-box[0])*(box[3]-box[1]));ratio=area/old_area
        upper=1.25 if target.get('motion_quality',0.)>=.75 else 1.6
        if .65<=ratio<=upper and _iou(box,old)>=.65:return True
        # Once the old extent is rejected, insisting on its size/overlap can
        # prevent a fresh, complete segmentation from correcting it. A current
        # learned identity plus measured endpoints may replace that suspect box.
        if target.get('ever_confirmed') and target.get('fail',0)>0 \
                and .5<=ratio<=1.6 and self._near(box,old,rel=.75) \
                and score>=self.args.scan_cos_th+.06:
            pattern=self._registered_pattern(box,target['cls'])
            reference=self._registered_appearance_similarity(box,target['cls'])
            if pattern is not None and pattern['similarity']>=.82 \
                    and self._registered_identity_ok(reference,pattern,target['cls']) \
                    and self._localized_geometry_ok(box,target['cls']) \
                    and (endpoint_boundary_support(self.frame,box)[1]
                         or measured_tapered_boundary(self.frame,box,self.data.class_ar(target['cls']))
                         or measured_narrow_tip(self.frame,box,self.data.class_ar(target['cls']))):
                return True
        # A recovered object can translate beyond the old box's IoU limit.
        # Re-establish its identity on current pixels instead of rejecting a
        # genuine displacement as a change of boundary. No old evidence renews.
        if target.get('ever_confirmed') and target.get('fail',0)>0 \
                and .8<=ratio<=1.25 and self._near(box,old,rel=.75) \
                and score>=self.args.scan_cos_th+.02 \
                and time.perf_counter()-target.get('strong_identity_time',0.)<=.6:
            pattern=self._registered_pattern(box,target['cls'])
            descriptor=describe_appearance(self.frame,box)
            similarity=max((float(descriptor@v) for v in target.get('appearance_bank',[])),default=0.) \
                if descriptor is not None else 0.
            if pattern is not None and pattern['similarity']>=.82 and similarity>=.8 \
                    and self._registered_appearance_similarity(box,target['cls'])>=.58 \
                    and endpoint_boundary_support(self.frame,box)[1]:
                return True
        if target.get('ever_confirmed') and 1.<ratio<=1.75 \
                and score>=self.args.scan_cos_th+.02 \
                and shell_extension_support(self.frame,old,box,self.data.class_ar(target['cls']))>0.:
            pattern=self._registered_pattern(box,target['cls'])
            if pattern is not None and pattern['similarity']>=.82 \
                    and self._registered_appearance_similarity(box,target['cls'])>=.55:
                return True
        # A zoomed-out object can occupy much less than its old tracking box.
        # Accept a tighter extent only with stronger CURRENT identity evidence.
        intersection=max(0,min(box[2],old[2])-max(box[0],old[0])) \
                     *max(0,min(box[3],old[3])-max(box[1],old[1]))
        return bool(target.get('ever_confirmed') and .25<=ratio<.65
                    and intersection/area>=.85
                    and score>=max(self.args.scan_cos_th+.02,float(target.get('cos',1.))+.08)
                    and self._registered_appearance_similarity(box,target['cls'])>=.62)

    def _refine_bright_boundaries(self,candidates,ci):
        """Recheck a few segmentation padding choices on the current image."""
        if not candidates or not self.data.class_visual_refs(ci):
            return candidates
        boxes=[box for box,occupancy in candidates]
        measured_key=(self.frame_idx,getattr(self,'_capture_time',None),self._search_epoch)
        if getattr(self,'_measured_boundary_key',None)!=measured_key:
            self._measured_boundary_key=measured_key
            self._measured_boundary_boxes=set()
        self._measured_boundary_boxes.update(boxes)
        scores=self._score_localized_boxes(boxes)[:,ci]
        references=[self._registered_appearance_similarity(box,ci) for box in boxes]
        # Already accepted strong crops do not need speculative padding. The
        # frame cache avoids repeating these CNN results in the caller.
        selected=[j for j in np.argsort(references)[::-1] if references[j]>=.52
                  and (scores[j]<self.args.scan_cos_th+.02 or references[j]<.62
                       or (self._generic_class(ci) and not self._new_white_identity_ok(boxes[j],ci,float(scores[j]))))][:3]
        extra=[];seen=set(boxes)
        original_verified=any(float(scores[j])>=self.args.scan_cos_th+.02
            and self._localized_geometry_ok(box,ci) and self._shape_gate_ok(box,ci,True)
            and self._new_white_identity_ok(box,ci,float(scores[j]))
            for j,box in enumerate(boxes))
        # A cut opaque seed can be a marginal learned match. Its reflected
        # shell is only a proposal: verify learned identity on the WHOLE new
        # box here, and re-run CNN/color/background/negative gates in callers.
        for j in ([] if original_verified else np.argsort(scores)[::-1][:2]):
            if scores[j]<self.args.scan_cos_th+.02 or references[j]<.50:
                continue
            pattern=self._registered_pattern(boxes[j],ci)
            if pattern is None or pattern['similarity']<.70:
                continue
            for shell in reflected_shell_candidates(self.frame,boxes[j],self.data.class_ar(ci)):
                if shell not in seen and self._localized_geometry_ok(shell,ci) \
                        and self._new_white_identity_ok(shell,ci,float(scores[j])):
                    seen.add(shell);extra.append((shell,candidates[j][1]))
        # Even a confidently recognized opaque body can omit a transparent tip.
        # Only measured paired rails may extend such an already good candidate.
        for j in np.argsort(references)[::-1][:2]:
            pattern=self._registered_pattern(boxes[j],ci)
            if references[j]<.58 or pattern is None or pattern['similarity']<.78:
                continue
            for shell in anchored_shell_candidates(self.frame,boxes[j],self.data.class_ar(ci))[:2]:
                if shell not in seen and self._new_white_identity_ok(shell,ci,self.args.scan_cos_th+.02):
                    seen.add(shell);extra.append((shell,candidates[j][1]))
        for j in selected:
            x1,y1,x2,y2=boxes[j];w,h=x2-x1,y2-y1
            variants=[(x1,y1,int(x2-.02*w),y2),
                      (x1,y1,min(self.W,x2+int(.05*w)),y2),
                      (x1,y1+int(.03*h),x2,y2),
                      (x1,y1+int(.05*h),int(x2-.02*w),y2)]
            pattern=self._registered_pattern(boxes[j],ci)
            if pattern is not None and pattern['similarity']>=.78:
                # A small bright seed may cut the gray/white outer shell. Try
                # bounded body corrections; each new crop must improve learned
                # appearance and pass all gates before it can be admitted.
                variants.extend([(x1,y1,min(self.W,x2+int(.10*w)),y2),
                                 (x1,y1,min(self.W,x2+int(.15*w)),y2),
                                 (max(0,x1-int(.10*w)),y1,x2,y2),
                                 (max(0,x1-int(.15*w)),y1,x2,y2)])
            for box in variants:
                if box in seen or not self._localized_geometry_ok(box,ci):
                    continue
                reference=self._registered_appearance_similarity(box,ci)
                # Strong registration supports correcting a rejected border;
                # weaker registration must actually improve when it is moved.
                supports=(references[j]>=.62 and reference>=references[j]-.005) \
                         or reference>=references[j]+.01
                if reference>=.58 and supports:
                    seen.add(box);extra.append((box,candidates[j][1]))
                if len(extra)>=8:
                    return candidates+extra[:8]
        return candidates+extra[:8]

    def _colour_boundary(self, box, ci):
        """Match an independently measured current component to this extent."""
        key=(self.frame_idx,getattr(self,'_capture_time',None),self._search_epoch)
        if getattr(self,'_colour_boundary_key',None)!=key:
            self._colour_boundary_key=key
            self._colour_boundary_cache={}
        cache_key=(ci,tuple(box))
        if cache_key not in self._colour_boundary_cache:
            x1,y1,x2,y2=box;w,h=x2-x1,y2-y1
            region=(x1-int(.35*w),y1-int(.35*h),x2+int(.35*w),y2+int(.35*h))
            fits=colour_component_candidates(self.frame,self.data.class_hist(ci),region)
            fit=max(fits,key=lambda item:_iou(box,item[0]),default=None)
            self._colour_boundary_cache[cache_key]=fit if fit is not None and _iou(box,fit[0])>=.75 else None
        return self._colour_boundary_cache[cache_key]

    def _accept_colour_target(self,tgt,box,score,source,boundary=False):
        """Opaque colour objects use their measured plane, not white tips/body."""
        ci=tgt['cls'];now=time.perf_counter()
        tgt['admission_pending']=False
        current=float(self._score_localized_boxes([box])[0,ci])
        self._diag_localized[ci]=self._last_localized_evidence[0][ci]
        required=self.args.scan_cos_th+(.02 if not tgt.get('ever_confirmed') else -.14)
        score=min(float(score),current)
        if self._generic_class(ci) and score<0.:
            # Hard feature/colour/background/negative vetoes cannot be rescued
            # by geometry. Avoid running registration for an unusable crop.
            reason=f'目标特征核验·{self._diag_localized[ci]["reason"]}({score:.0%})'
            tgt.update(verified=False,coasting=False,pending_reason=reason)
            if tgt.get('identity_fail_frame')!=self.frame_idx:
                tgt['fail']=int(tgt.get('fail',0))+1
                tgt['identity_fail_frame']=self.frame_idx
            self._diag_reject[ci]=reason
            return False
        has_refs=bool(self.data.class_visual_refs(ci))
        pattern=self._registered_pattern(box,ci) if has_refs else None
        reference=self._registered_appearance_similarity(box,ci) if has_refs else 1.
        identity=(not has_refs and not self._generic_class(ci)) or self._registered_identity_ok(reference,pattern,ci) or bool(
            self._geometry_identity_ok(pattern,ci)
            and reference>=.65 and score>=self.args.scan_cos_th+.04)
        chromatic=chromatic_model(self.data.class_hist(ci)) is not None
        fit=self._colour_boundary(box,ci) if chromatic and not self._generic_class(ci) else None
        reason=None
        geometric_admission=self._generic_geometry_admission(box,ci,score)
        if score<required and not geometric_admission:reason=f'目标特征核验·{self._diag_localized[ci]["reason"]}({score:.0%})'
        elif not identity:reason=(f'彩色目标身份不符·外观{reference:.0%}、内部结构'
                                 +(f'{pattern["similarity"]:.0%}' if pattern is not None else '无参考'))
        elif chromatic and fit is None and not self._generic_class(ci):reason='彩色目标边界与当前颜色组件不符'
        elif fit is None and not self._localized_geometry_ok(box,ci):reason='目标几何与学习区域不符'
        elif min(box[2]-box[0],box[3]-box[1])<16 or not self._shape_gate_ok(box,ci,False):reason='彩色目标内部结构不足'
        elif not self._cs_gate_ok(box,cs_th=.16):reason='彩色目标与周围太像'
        elif not self._generic_class(ci) and self._skin_ratio(box)>.65:reason='彩色目标框内肤色过多'
        if reason is not None:
            if self._generic_class(ci) and not identity and score>=required and self._identity_pending(ci):
                tgt.update(verified=False,coasting=False,pending_reason='身份核验待继续')
                self._diag_reject[ci]='身份核验待继续'
                return False
            tgt.update(verified=False,coasting=False,pending_reason=reason)
            if tgt.get('identity_fail_frame')!=self.frame_idx:
                tgt['fail']=int(tgt.get('fail',0))+1;tgt['identity_fail_frame']=self.frame_idx
            self._diag_reject[ci]=reason
            return False
        # Each independently segmented camera frame may confirm once. Neither
        # a template nor an old worker result counts as a new colour boundary.
        if fit is not None and tgt.get('colour_boundary_frame')!=self.frame_idx:
            tgt['boundary_votes']=int(tgt.get('boundary_votes',0))+1
            tgt['colour_boundary_frame']=self.frame_idx
        if tgt.get('last_evidence_frame')!=self.frame_idx:
            limit=3 if fit is not None and tgt.get('boundary_votes',0)<2 else 4
            tgt['pend']=min(limit,int(tgt.get('pend',0))+1)
        boundary_age=int(tgt.get('boundary_age',0)) if self._generic_class(ci) else 0
        if self._generic_class(ci) and boundary and not getattr(self,'_generic_tracking',False):
            # Only an accepted independent localization refreshes this age.
            # LK/current-identity success alone does not measure a new extent.
            boundary_age=0
            tgt['generic_boundary_age_frame']=self.frame_idx
            tgt['generic_boundary_time']=now
        tgt.update(box=tuple(box),ema=None,asize=(box[2]-box[0],box[3]-box[1]),
            cos=score,gcos=score,verified=True,coasting=False,fail=0,pending_reason='',
            last_evidence_frame=self.frame_idx,last_verified_time=now,
            ever_confirmed=bool(tgt.get('ever_confirmed') or tgt.get('pend',0)>=4),
            registered_identity_similarity=reference,
            registered_pattern_similarity=None if pattern is None else round(pattern['similarity'],3),
            registered_dense_alignment=None if pattern is None else pattern.get('dense'),
            registered_spatial_alignment=None if pattern is None else pattern.get('spatial'),
            identity_evidence_frame=self.frame_idx,identity_blocked=False,
            boundary_source='colour_component' if fit is not None else 'generic',
            colour_boundary=fit is not None,boundary_age=boundary_age)
        tgt['generic_geometry_admission']=geometric_admission
        if self._generic_class(ci):
            self._remember_generic_hint(ci,pattern)
            tgt.setdefault('generic_boundary_time',now)
            # Recovery privileges never authorize a displayed stale identity.
            strong=bool(score>=self.args.scan_cos_th+.02 and pattern is not None
                        and (pattern['similarity']>=.78 or geometric_admission))
            if strong:tgt['strong_identity_time']=now
            recent_strong=now-tgt.get('strong_identity_time',0.)<=.6
            tgt['generic_recovery_until']=now+(.60 if recent_strong else .40 if tgt['ever_confirmed'] else .25)
            tgt['generic_identity_box']=tuple(box)
            tgt['generic_anchor_appearance']=describe_appearance(self.frame,box)
            tgt['generic_hidden_motion']=False
            tgt.pop('generic_recovery_box',None)
        if fit is not None:
            tgt['colour_boundary_evidence']=fit[1]
            tgt['segmentation_frame']=self.frame_idx;tgt['segmentation_time']=now
        if 'motion_gray' not in tgt or (self._generic_class(ci)
                and not getattr(self,'_generic_tracking',False)):
            self._init_target_motion(tgt)
        self._diag_reject[ci]=f'{source}·彩色边界复核{score:.0%}'
        return True

    def _track_generic_target(self,tgt):
        started=time.perf_counter()
        previous=getattr(self,'_generic_identity_fast',False)
        tracking=getattr(self,'_generic_tracking',False)
        self._generic_identity_fast=True
        self._generic_tracking=True
        try:
            if tgt.get('generic_boundary_age_frame')!=self.frame_idx:
                tgt['boundary_age']=int(tgt.get('boundary_age',0))+1
                tgt['generic_boundary_age_frame']=self.frame_idx
            self._track_generic_target_fast(tgt)
        finally:
            self._generic_identity_fast=previous
            self._generic_tracking=tracking
            perf=getattr(self,'_generic_perf',{})
            perf['generic_track_ms']=perf.get('generic_track_ms',0.)+(time.perf_counter()-started)*1000.
            self._generic_perf=perf

    def _seed_generic_motion(self,box,ci,gray):
        pattern=self._registered_pattern(box,ci)
        if pattern is not None:
            points=pattern.get('motion_points')
            if points is not None and len(points)>=6:
                x,y,r,b=box
                result=np.float32(points)*[(r-x)/96.,(b-y)/96.]+[x,y]
                inside=(result[:,0]>=x)&(result[:,0]<r)&(result[:,1]>=y)&(result[:,1]<b)
                if inside.sum()>=6:return result[inside].astype(np.float32).reshape(-1,1,2)
        return seed_points(self.frame,gray,box,prefer_bright=False)

    def _supported_generic_motion(self,motion,base):
        # A localized point cluster cannot determine rotation/scale of the
        # entire learned ROI. Keep only its measured translation as a proposal;
        # all current identity, feature and display gates still apply.
        if motion is not None and (motion.coverage[0]<.2 or motion.coverage[1]<.25):
            box=shift_box(base,motion.velocity,self.W,self.H)
            # Preserve small, locally consistent transforms. Only discard a
            # whole-ROI extrapolation whose edges depart substantially from
            # the displacement actually measured on the supported points.
            tolerance=max(2.,.10*min(base[2]-base[0],base[3]-base[1]))
            if max(abs(a-b) for a,b in zip(box,motion.box))<=tolerance:return motion
            dx,dy=box[0]-base[0],box[1]-base[1]
            x,y,r,b=base
            corners=np.float64([[x+dx,y+dy],[r+dx,y+dy],[r+dx,b+dy],[x+dx,b+dy]])
            span=np.ptp(motion.points.reshape(-1,2),axis=0)
            coverage=(float(span[0]/max(1,box[2]-box[0])),float(span[1]/max(1,box[3]-box[1])))
            motion=motion._replace(box=box,corners=corners,coverage=coverage)
        return motion

    def _predict_generic_motion(self,tgt):
        key=(self.frame_idx,getattr(self,'_capture_time',None),self._search_epoch)
        if getattr(self,'_generic_motion_cache_key',None)!=key:
            self._generic_motion_cache_key=key
            self._generic_motion_cache={}
            self._generic_motion_gray=cv2.cvtColor(self.frame,cv2.COLOR_BGR2GRAY)
        if id(tgt) in self._generic_motion_cache:
            return self._generic_motion_cache[id(tgt)]
        gray=self._generic_motion_gray
        base=tgt.get('motion_box',tgt['box'])
        motion=track_points(tgt.get('motion_gray'),gray,tgt.get('motion_points'),
                            base,tgt.get('motion_corners'))
        motion=self._supported_generic_motion(motion,base)
        if motion is not None:
            capture=getattr(self,'_capture_time',None)
            reference=tgt.get('motion_capture_time')
            fps=max(5.,float(getattr(self,'_perf_current',{}).get('camera_fps',30.)))
            steps=max(1.,(capture-reference)*fps) if capture is not None and reference is not None else 1.
            motion=motion._replace(velocity=tuple(v/steps for v in motion.velocity))
        box=motion.box if motion is not None else base
        tgt['motion_quality']=0. if motion is None else motion.inlier_ratio
        tgt['motion_coverage']=None if motion is None else motion.coverage
        tgt['motion_point_count']=0 if motion is None else len(motion.points)
        now=time.perf_counter();relocation=None
        if (motion is None and tgt.get('ever_confirmed')
                and now<=tgt.get('generic_recovery_until',0.)):
            search=recovery_region(base,tgt.get('velocity',(0.,0.)),self.W,self.H,tgt.get('fail',0))
            relocation=relocate_template(tgt.get('motion_gray'),gray,base,
                min_similarity=.82,search_box=search)
            if relocation is not None:box=relocation[0]
        # A failed hidden acquisition may have found a position last frame.
        # Recheck its CURRENT pixels; never carry identity or confirmation.
        source=tgt.get('_generic_acquisition_source')
        if source is not None and (source['epoch']!=self._search_epoch
                or not 0.<=now-source['time']<=.6 or tgt.get('ever_confirmed')):
            tgt.pop('_generic_acquisition_source',None);source=None
        if (source is not None and motion is None and not tgt.get('ever_confirmed')
                and now<=tgt.get('generic_recovery_until',0.)
                and 0.<=now-tgt.get('generic_recovery_box_time',0.)<=.15):
            box=tuple(tgt.get('generic_recovery_box',box))
        tgt['generic_relocation_similarity']=None if relocation is None else relocation[1]
        prediction=(gray,motion,box,relocation)
        self._generic_motion_cache[id(tgt)]=prediction
        return prediction

    def _prefetch_generic_tracks(self):
        """Batch current-frame feature gates; do not grant identity or votes."""
        targets=[t for t in self.scan_targets if self._generic_class(t['cls'])]
        if len(targets)<2:
            return
        started=time.perf_counter()
        boxes=[self._predict_generic_motion(t)[2] for t in targets]
        previous=getattr(self,'_generic_identity_fast',False)
        self._generic_identity_fast=True
        try:
            self._score_localized_boxes(boxes)
        finally:
            self._generic_identity_fast=previous
            perf=getattr(self,'_generic_perf',{})
            perf['generic_batch_ms']=(time.perf_counter()-started)*1000.
            perf['generic_batch_targets']=len(targets)
            self._generic_perf=perf

    def _reanchor_generic_track(self,tgt,box):
        """Propose a bounded learned ROI correction, then remeasure its pixels.

        Fine correspondences may propose this crop but never authorize it.
        A failed/unfinished correction must keep the original current proof.
        """
        ci=tgt['cls']
        if (not tgt.get('ever_confirmed') or self.frame_idx-tgt.get('generic_reanchor_frame',-1000)<6
                or getattr(self,'_identity_deadlines',{}).get(ci,0.)-time.perf_counter()<.004):
            return box
        pattern=self._registered_pattern(box,ci);spatial={} if pattern is None else pattern.get('spatial',{})
        if (spatial.get('patch_size')!=16 or spatial.get('inliers',0)<16
                or not .78<=spatial.get('extent',0.)<.90):return box
        quad=np.asarray(pattern.get('full_quad'),np.float32)
        if quad.shape!=(4,2) or not np.isfinite(quad).all() or not cv2.isContourConvex(quad):return box
        x,y,r,b=box;quad=quad*[(r-x)/96.,(b-y)/96.]+[x,y]
        lo,hi=quad.min(axis=0),quad.max(axis=0)
        fitted=(max(0,int(np.floor(lo[0]))),max(0,int(np.floor(lo[1]))),
                min(self.W,int(np.ceil(hi[0]))),min(self.H,int(np.ceil(hi[1]))))
        change=max(abs(a-b) for a,b in zip(fitted,box));side=min(r-x,b-y)
        if (min(fitted[2]-fitted[0],fitted[3]-fitted[1])<24 or _iou(fitted,box)<.75
                or not max(2.,.03*side)<=change<=.15*side):return box
        tgt['generic_reanchor_frame']=self.frame_idx
        # The primary proof remains valid on its own crop. Speculative task
        # state and diagnostics must not replace it if this new crop fails.
        saved={}
        for name in ('_identity_task_states','_generic_identity_hint_bank','_generic_identity_hints',
                     '_diag_reject','_diag_localized'):
            mapping=getattr(self,name)
            value=mapping.get(ci)
            saved[name]=(ci in mapping,copy.deepcopy(value) if name=='_identity_task_states' else value)
        accepted=False
        try:
            score=float(self._score_localized_boxes([fitted])[0,ci])
            current=self._registered_pattern(fitted,ci) if score>=0. else None
            measured={} if current is None else current.get('spatial',{})
            # A correction must preserve the learned view/pose that proposed
            # it. A different view may prove identity, but cannot validate the
            # old view's projected boundaries. Normal view selection stays.
            pose=lambda proof:tuple(proof.get(k) for k in ('reference','flip','rotation'))
            if measured.get('patch_size')!=16 or pose(measured)!=pose(spatial):return box
            extent=measured.get('extent',0.)
            if extent<spatial['extent']+.03:return box
            trial=dict(tgt)
            if not self._accept_colour_target(trial,fitted,score,'学习参考当前框校正'):return box
            # _accept_colour_target updates mapping entries; motion arrays are
            # read-only here. Keep target identity for the caller's track list.
            tgt.clear();tgt.update(trial)
            getattr(self,'_generic_motion_cache',{}).pop(id(tgt),None)
            accepted=True
            return fitted
        finally:
            if not accepted:
                for name,(present,value) in saved.items():
                    mapping=getattr(self,name)
                    if present:mapping[ci]=value
                    else:mapping.pop(ci,None)

    def _track_generic_target_fast(self,tgt):
        """Track any learned appearance, rechecking current identity each frame."""
        ci=tgt['cls']
        gray,motion,box,relocation=self._predict_generic_motion(tgt)
        with self._generic_reference_trial(ci) as commit_reference:
            score=float(self._score_localized_boxes([box])[0,ci])
            accepted=self._accept_colour_target(tgt,box,score,'通用物体当前帧复核')
            if accepted:commit_reference()
        if accepted:
            fitted=self._reanchor_generic_track(tgt,box)
            if tuple(fitted)!=tuple(box):
                box=fitted;motion=None  # Reseed real corners in the corrected CURRENT crop.
            tgt['motion_gray']=gray
            tgt['motion_capture_time']=getattr(self,'_capture_time',None)
            tgt['motion_box']=tuple(box)
            if motion is not None:
                tgt['motion_points']=motion.points
                tgt['motion_corners']=motion.corners
                tgt['velocity']=motion.velocity
            if motion is None or self.frame_idx%12==0:
                tgt['motion_points']=self._seed_generic_motion(box,ci,gray)
            if motion is None:
                x,y,r,b=box
                tgt['motion_corners']=np.float32([[x,y],[r,y],[r,b],[x,b]])
            return
        tgt.update(verified=False,coasting=False)
        now=time.perf_counter()
        until=tgt.get('generic_recovery_until',tgt.get('last_verified_time',0.)+.4)
        source=tgt.get('_generic_acquisition_source')
        if (not tgt.get('ever_confirmed') and motion is None and source is not None
                and source['epoch']==self._search_epoch and 0.<=now-source['time']<=.6
                and now<=until and tgt.get('pend',0)>0):
            # Keep the original independently localized source separate from
            # advancing LK frames. A template proposes only; next frame runs
            # the same feature, identity and confirmation gates on its crop.
            recovered=relocate_template(source['gray'],gray,source['box'],min_similarity=.82,
                search_box=recovery_region(tgt['box'],tgt.get('velocity',(0.,0.)),self.W,self.H,tgt.get('fail',0)))
            if recovered is not None:
                tgt['generic_recovery_box']=tuple(recovered[0])
                tgt['generic_recovery_box_time']=getattr(self,'_capture_time',None) or now
        if relocation is not None and now<=until and score>=0.:
            anchor=tgt.get('generic_anchor_appearance');current=describe_appearance(self.frame,box)
            if (anchor is not None and current is not None and float(anchor@current)>=.65
                    and self._registered_appearance_similarity(box,ci)>=.58):
                # A current search hint, not a new optical-flow state or identity.
                tgt['generic_recovery_box']=tuple(box)
                tgt['generic_recovery_box_time']=now
        # Follow reliable geometry internally during a bounded identity gap.
        # Do not update the identity anchor, confirmation count or deadline.
        if motion is not None and tgt.get('ever_confirmed') and now<=until:
            anchor=tgt.get('generic_anchor_appearance')
            current=describe_appearance(self.frame,box)
            identity_box=tgt.get('generic_identity_box',tgt['box'])
            ratio=((box[2]-box[0])*(box[3]-box[1]))/max(1.,
                   (identity_box[2]-identity_box[0])*(identity_box[3]-identity_box[1]))
            reliable=(motion.inlier_ratio>=.70 and len(motion.points)>=8
                and min(motion.coverage)>=.25 and .6<=ratio<=1.6 and score>=0.
                and anchor is not None and current is not None and float(anchor@current)>=.65
                and self._registered_appearance_similarity(box,ci)>=.58)
            if reliable:
                tgt.update(motion_box=tuple(box),motion_gray=gray,motion_points=motion.points,
                    motion_corners=motion.corners,velocity=motion.velocity,
                    motion_capture_time=getattr(self,'_capture_time',None),
                    generic_hidden_motion=True,generic_hidden_motion_time=now)
        if now>until:
            if tgt in self.scan_targets:self.scan_targets.remove(tgt)

    def _track_colour_target(self,tgt):
        ci=tgt['cls'];x1,y1,x2,y2=tgt['box'];w,h=x2-x1,y2-y1
        region=(x1-int(.35*w),y1-int(.35*h),x2+int(.35*w),y2+int(.35*h))
        fits=colour_component_candidates(self.frame,self.data.class_hist(ci),region)
        fits=sorted(fits,key=lambda fit:_iou(fit[0],tgt['box']),reverse=True)
        for box,evidence in fits:
            if not self._near(box,tgt['box'],rel=.75):continue
            score=float(self._score_localized_boxes([box])[0,ci])
            if self._accept_colour_target(tgt,box,score,'颜色组件跟踪'):
                return
        tgt.update(verified=False,coasting=False)
        if tgt.get('identity_fail_frame')!=self.frame_idx:
            tgt['fail']=int(tgt.get('fail',0))+1;tgt['identity_fail_frame']=self.frame_idx
        if not fits:self._diag_reject[ci]='未找到当前彩色目标边界'
        if tgt.get('fail',0)>6 or time.perf_counter()-tgt.get('last_verified_time',0.)>.5:
            if tgt in self.scan_targets:self.scan_targets.remove(tgt)

    def _measured_generic_extent(self,box,pattern,min_overlap=.6):
        """Project the whole LEARNED ROI, never an arbitrary material blob."""
        if pattern is None or pattern.get('full_quad') is None:return None
        # Fine correspondences establish identity in this crop. Their affine
        # fit is not an independent edge measurement: keep the verified ROI
        # until a stronger boundary method can justify replacing its edges.
        if pattern.get('spatial',{}).get('patch_size')==16:return None
        if not any(pattern.get(k,{}).get('verified') for k in ('geometry','dense','spatial')):return None
        quad=np.asarray(pattern['full_quad'],np.float32)
        if quad.shape!=(4,2) or not np.isfinite(quad).all() or not cv2.isContourConvex(quad):return None
        x,y,r,b=box
        quad=quad*[(r-x)/96.,(b-y)/96.]+[x,y]
        lo,hi=quad.min(axis=0),quad.max(axis=0)
        fitted=(max(0,int(np.floor(lo[0]))),max(0,int(np.floor(lo[1]))),
                min(self.W,int(np.ceil(hi[0]))),min(self.H,int(np.ceil(hi[1]))))
        if min(fitted[2]-fitted[0],fitted[3]-fitted[1])<24 or _iou(fitted,box)<min_overlap:return None
        if max(abs(a-b) for a,b in zip(fitted,box))<max(2,.03*min(r-x,b-y)):return None
        return fitted,quad

    def _cache_measured_geometry(self,box,fitted,quad,ci,pattern):
        """Express this frame's measured homography in a corrected crop.

        Re-cropping the SAME pixels does not require matching the same points
        again. Feature/colour/background/negative and appearance checks still
        run on the corrected crop; no proof survives a frame-key change.
        """
        if not pattern.get('geometry',{}).get('verified'):return
        key=(self.frame_idx,self._search_epoch)
        if getattr(self,'_pattern_cache_key',None)!=key:return
        x,y,r,b=box;fx,fy,fr,fb=fitted
        def relative(points):
            absolute=np.asarray(points,np.float32)*[(r-x)/96.,(b-y)/96.]+[x,y]
            return ((absolute-[fx,fy])*[96./(fr-fx),96./(fb-fy)]).tolist()
        normalized=(quad-[fx,fy])*[96./(fr-fx),96./(fb-fy)]
        lo,hi=quad.min(axis=0),quad.max(axis=0)
        intersection=float(np.prod(np.maximum(0,np.minimum(hi,[fr,fb])-np.maximum(lo,[fx,fy]))))
        extent=intersection/max(1.,float(np.prod(hi-lo))+(fr-fx)*(fb-fy)-intersection)
        current={k:v for k,v in pattern.items() if k.startswith('_hint_')}
        if current.get('_hint_dense_seed') is not None:
            seed=dict(current['_hint_dense_seed'])
            seed_quad=np.asarray(seed.get('full_quad',[]),np.float32)
            if seed_quad.shape==(4,2) and np.isfinite(seed_quad).all():
                seed['full_quad']=relative(seed_quad);current['_hint_dense_seed']=seed
            else:current.pop('_hint_dense_seed',None)
        current.update(similarity=pattern.get('similarity',-1.),
            geometry=dict(pattern['geometry'],extent_iou=extent),full_quad=normalized.tolist(),
            patch=tuple(normalized.min(axis=0))+tuple(np.ptp(normalized,axis=0)))
        if pattern.get('motion_points') is not None:current['motion_points']=relative(pattern['motion_points'])
        self._pattern_cache[(ci,tuple(fitted))]=current
        getattr(self,'_localized_score_cache',{}).pop(tuple(fitted),None)

    def _cached_generic_reference_fit(self,box,ci):
        """Propose learned bounds from THIS frame's existing correspondences.

        A padded material component may fail identity because its edges differ
        from the registered ROI. That does not invalidate the measured mapping.
        This only supplies a new crop: callers must score and verify it again.
        No matching is started and no evidence is carried across frames.
        """
        if not self._generic_class(ci):return None
        if getattr(self,'_pattern_cache_key',None)!=(self.frame_idx,self._search_epoch):return None
        pattern=self._pattern_cache.get((ci,tuple(box)))
        if pattern is None or not pattern.get('geometry',{}).get('verified'):return None
        fitted=self._measured_generic_extent(box,pattern,min_overlap=.25)
        if fitted is not None:
            self._cache_measured_geometry(box,fitted[0],fitted[1],ci,pattern)
        return fitted

    def _fit_generic_reference(self,box,ci,allow_projection=True):
        if not self._generic_class(ci):return None
        if not self._foreground_generic_identity():
            fitted=self._cached_generic_reference_fit(box,ci)
            if fitted is not None:return fitted
        # Verify first. Accurate whole-reference alignment is cheaper and safer
        # than repeatedly rebuilding an already correct box from sparse points.
        pattern=self._registered_pattern(box,ci)
        if self._registered_identity_ok(self._registered_appearance_similarity(box,ci),pattern,ci):
            # Current foreground identity is already established in this ROI.
            # An optional geometric projection must not replace that verified
            # crop before tracking confirmation; partial learned ROIs are valid.
            # Failed crops below still use the established repair path.
            if self._foreground_generic_identity():return None
            # Identity on a padded candidate does not imply its edges are the
            # learned boundaries. Caller must recheck the fitted CURRENT crop.
            fitted=self._measured_generic_extent(box,pattern)
            if fitted is not None and pattern.get('geometry',{}).get('verified'):
                self._cache_measured_geometry(box,fitted[0],fitted[1],ci,pattern)
                return fitted
            return None if self._foreground_generic_identity() else fitted
        if not allow_projection and self._foreground_generic_identity():
            return None
        x1,y1,x2,y2=box;w,h=x2-x1,y2-y1
        region=(max(0,int(x1-.2*w)),max(0,int(y1-.2*h)),
                min(self.W,int(x2+.2*w)),min(self.H,int(y2+.2*h)))
        left,top,right,bottom=region
        match=match_colour_geometry(self.frame[top:bottom,left:right],
                                    self.data.class_colour_geometry_refs(ci))
        if match is not None and match.get('full_quad') is not None:
            quad=np.asarray(match['full_quad'],np.float32)*[(right-left)/96.,(bottom-top)/96.]+[left,top]
            lo,hi=quad.min(axis=0),quad.max(axis=0)
            fitted=(max(0,int(np.floor(lo[0]))),max(0,int(np.floor(lo[1]))),
                    min(self.W,int(np.ceil(hi[0]))),min(self.H,int(np.ceil(hi[1]))))
            if min(fitted[2]-fitted[0],fitted[3]-fitted[1])>=24:
                return fitted,quad
        # An independently verified current extent needs no speculative repair.
        # In particular, sparse internal features must not expand a valid crop
        # into the hand/background just because a second projection is possible.
        if self._new_white_identity_ok(box,ci,1.):
            return None
        if not allow_projection:
            # The worker owns full-resolution ORB/SIFT recovery. A failed
            # current foreground check waits for a fresh verified proposal.
            return None
        candidates=self._project_generic_references(self.frame[top:bottom,left:right],ci)
        if candidates:
            # Projection supplies the complete learned extent, including parts
            # with no keypoints. Prefer the current target, not another instance.
            fits=[]
            for candidate,proof in candidates:
                fitted=tuple(v+(left if j%2==0 else top) for j,v in enumerate(candidate))
                if _iou(fitted,box)<.25:continue
                quad=np.asarray(proof['full_quad'],np.float32)+[left,top]
                fits.append((_iou(fitted,box),fitted,quad))
            if fits:
                _,fitted,quad=max(fits,key=lambda item:item[0])
                # The same current-frame correspondences already establish
                # identity for the fitted crop. Re-extracting SIFT/ORB in that
                # crop doubles the work and can choose inconsistent points.
                matched=next(proof for candidate,proof in candidates
                    if tuple(v+(left if j%2==0 else top) for j,v in enumerate(candidate))==fitted)
                lo,hi=quad.min(axis=0),quad.max(axis=0)
                fl=np.float32(fitted[:2]);fh=np.float32(fitted[2:])
                intersection=float(np.prod(np.maximum(0,np.minimum(hi,fh)-np.maximum(lo,fl))))
                extent=intersection/max(1.,float(np.prod(hi-lo)+np.prod(fh-fl))-intersection)
                proof=dict(matched,extent_iou=round(extent,3))
                relative=(quad-fl)*[96./(fh[0]-fl[0]),96./(fh[1]-fl[1])]
                lower,upper=relative.min(axis=0),relative.max(axis=0)
                key=(self.frame_idx,self._search_epoch)
                if getattr(self,'_pattern_cache_key',None)!=key:
                    self._pattern_cache_key=key;self._pattern_cache={}
                self._pattern_cache[(ci,fitted)]=dict(similarity=-1.,geometry=proof,
                    full_quad=relative.tolist(),patch=tuple(lower)+tuple(upper-lower))
                return fitted,quad
        return None

    def _generic_bright_candidates(self,ci,region=None):
        histogram=self.data.class_hist(ci)
        if histogram is None:return []
        hist=np.asarray(histogram,np.float32).reshape(8,8,8)
        hist=hist/max(float(hist.sum()),1e-9)
        if float(hist[:,0,:].sum())<.6 or float(hist[:,:,5:].sum())<.4:return []
        candidates=bright_component_candidates(self.frame,region or (0,0,self.W,self.H),
            self.data.class_ar(ci),accept_seed=lambda b:self._bright_seed_has_structure(b,ci),
            max_candidates=8)
        candidates=sorted(candidates,key=lambda item:self._registered_appearance_similarity(item[0],ci),reverse=True)
        candidates=[item for item in candidates if self._registered_appearance_similarity(item[0],ci)>=.58][:3]
        proposals=[];seen=set()
        for box,occupancy in candidates:
            x,y,r,b=box;w,h=r-x,b-y
            variants=[box]
            contour,_=bright_body_contour(self.frame[y:b,x:r])
            if contour is not None:
                bx,by,bw,bh=cv2.boundingRect(contour)
                # The visible connected component anchors both possible
                # extents. Whole learned appearance, not a preferred side or
                # transparent-tip rule, decides which proposal is the object.
                if w>=h:
                    left=x+round(bx*w/96);right=x+round((bx+bw)*w/96)
                    variants.extend([(max(0,right-w),y,right,b),(left,y,min(self.W,left+w),b)])
                else:
                    top=y+round(by*h/64);bottom=y+round((by+bh)*h/64)
                    variants.extend([(x,max(0,bottom-h),r,bottom),(x,top,r,min(self.H,top+h))])
            for variant in variants:
                if variant in seen or min(variant[2]-variant[0],variant[3]-variant[1])<24:continue
                seen.add(variant)
                proposals.append((variant,dict(component_occupancy=round(float(occupancy),3))))
        return proposals

    def _project_generic_references(self,frame,ci,min_extent=0.):
        # ORB is cheap; scale-space features are a second independent proposal
        # source when small markings change scale/rotation. Both use the same
        # distributed RANSAC and complete-extent checks, with no category rule.
        for method in ('orb','sift'):
            remaining=getattr(self,'_generic_job_deadline',float('inf'))-time.perf_counter()
            if remaining<=0:break
            references=self.data.class_scene_refs(ci,method)
            remaining=getattr(self,'_generic_job_deadline',float('inf'))-time.perf_counter()
            if remaining<=0:break
            worker=getattr(self,'_in_search_worker',False)
            states=getattr(self,'_generic_scene_states',{})
            candidates=project_reference_candidates(frame,references,method,
                time_budget=remaining if worker else None,
                state=states.setdefault(('repair',ci,method),{}) if worker else None)
            self._generic_scene_states=states
            candidates=[item for item in candidates if item[1]['extent_iou']>=min_extent]
            if candidates:return candidates
        return []

    def _generic_region_candidates(self,ci,region):
        key=(self.frame_idx,self._search_epoch,tuple(region))
        if getattr(self,'_region_candidates_key',None)!=key:
            x,y,r,b=region
            self._region_candidates_key=key
            self._region_candidates=[((box[0]+x,box[1]+y,box[2]+x,box[3]+y),proof)
                for box,proof in object_region_candidates(self.frame[y:b,x:r])]
        ranked=sorted(((self._registered_appearance_similarity(box,ci),box,proof)
                       for box,proof in self._region_candidates),key=lambda item:item[0],reverse=True)
        return [(box,proof) for similarity,box,proof in ranked[:6] if similarity>=.65]

    def _generic_expanded_candidates(self,ci,region):
        # Only reached after the original acquisition sources failed. Keep
        # expansion out of the latency of an already successful fast source.
        if not hasattr(self,'_generic_expanded_cursor'):self._generic_expanded_cursor={}
        cursor=self._generic_expanded_cursor.get(ci,0)
        self._generic_expanded_cursor[ci]=cursor+1
        if cursor%2:
            return self._generic_grid_candidates(ci,region,expanded=True)
        self._generic_region_candidates(ci,region)
        return self._generic_refined_candidates(ci,self._region_candidates)

    def _generic_refined_candidates(self,ci,candidates):
        """Keep complementary rankings and refine a bounded set of seeds."""
        if not candidates:return []
        boxes=list(dict.fromkeys(tuple(box) for box,_ in candidates))
        proofs={tuple(box):proof for box,proof in candidates}
        scores=self._generic_proposal_scores(boxes,ci)
        appearances=[self._registered_appearance_similarity(box,ci) for box in boxes]
        registration=registration_correlations(self.frame,boxes,self.data.class_dense_refs(ci))
        eligible=np.where(scores>=self.args.scan_cos_th+.02,scores,-1.)
        seeds=select_diverse(boxes,eligible,appearances,self.W,self.H,limit=6,registrations=registration)
        extra=[];seen=set(boxes)
        for i in seeds[:4]:
            for box in local_variants(boxes[i],self.W,self.H):
                if box not in seen:
                    seen.add(box);extra.append(box)
                    proofs[box]=dict(proofs[boxes[i]],refined_from=boxes[i])
        if extra:
            extra_scores=self._generic_proposal_scores(extra,ci)
            scores=np.concatenate([scores,extra_scores])
            appearances.extend(self._registered_appearance_similarity(box,ci) for box in extra)
            registration=np.concatenate([registration,
                registration_correlations(self.frame,extra,self.data.class_dense_refs(ci))])
            boxes.extend(extra)
        eligible=np.where(scores>=self.args.scan_cos_th+.02,scores,-1.)
        selected=select_diverse(boxes,eligible,appearances,self.W,self.H,limit=16,registrations=registration)
        return [(boxes[i],proofs[boxes[i]]) for i in selected]

    def _generic_proposal_scores(self,boxes,ci):
        # Cheap proposal ranking only. _search_generic_proposals rechecks the
        # selected windows with the unchanged three-view score and identity.
        crops=np.stack([crop_resize(self.frame,b,self.in_size) for b in boxes])
        features=self.backbone(preprocess_batch(crops))
        scores=self.clf.cosine_batch(features,self.data)
        return self._neg_gate(features,self._bg_gate(features,
            self._hist_gate(crops,scores)))[:,ci]

    def _generic_texture_candidates(self,ci,region,deadline):
        """Locate current learned texture before falling back to coarse grids."""
        refs=self.data.class_spatial_refs(ci,patch_size=16)
        if not refs:return []
        cursors=getattr(self,'_generic_texture_cursors',{})
        view=cursors.get(ci,0)%max(1,len(self.data.class_reference_views(ci)))
        cursors[ci]=view+1;self._generic_texture_cursors=cursors
        selected=[ref for ref in refs if ref['reference']==view]
        side=self.data.class_scan_fracs()[ci]*min(self.W,self.H);ar=self.data.class_ar(ci)
        size=(side*min(1.,ar),side*min(1.,1./ar))
        x,y,r,b=region
        found=spatial_scene_candidates(self.frame[y:b,x:r],selected,size,
            deadline=min(deadline,time.perf_counter()+.10))
        out=[]
        for box,pattern in found:
            current=(box[0]+x,box[1]+y,box[2]+x,box[3]+y)
            key=(self.frame_idx,self._search_epoch)
            if getattr(self,'_pattern_cache_key',None)!=key:
                self._pattern_cache_key=key;self._pattern_cache={}
            self._pattern_cache[(ci,current)]=pattern
            out.append((current,dict(texture_proposal=True,reference=view)))
        return out

    def _generic_grid_candidates(self,ci,region,expanded=False):
        left,top,right,bottom=region;w,h=right-left,bottom-top
        name='_generic_expanded_grid_cursor' if expanded else '_generic_grid_cursor'
        if not hasattr(self,name):setattr(self,name,{})
        cursors=getattr(self,name)
        cursor=cursors.get(ci,0)
        scales=self.args.scan_scales
        selected=[scales[(cursor+j)%len(scales)] for j in range(min(3,len(scales)))]
        cursors[ci]=(cursor+len(selected))%len(scales)
        boxes=[]
        for factor in selected:
            side=round(self.data.class_scan_fracs()[ci]*min(self.W,self.H)*factor)
            if not 24<=side<=max(w,h):continue
            boxes.extend((x+left,y+top,r+left,b+top)
                         for x,y,r,b in build_scan_boxes(w,h,side,48 if expanded else 24,ar=self.data.class_ar(ci)))
        boxes=list(dict.fromkeys(boxes))
        if not boxes:return []
        if expanded:
            return self._generic_refined_candidates(ci,[(box,dict(grid_proposal=True)) for box in boxes])
        crops=np.stack([crop_resize(self.frame,b,self.in_size) for b in boxes])
        features=self.backbone(preprocess_batch(crops))
        scores=self.clf.cosine_batch(features,self.data)
        scores=self._neg_gate(features,self._bg_gate(features,self._hist_gate(crops,scores)))[:,ci]
        ranked=sorted((j for j in range(len(boxes)) if scores[j]>=self.args.scan_cos_th+.02),
            key=lambda j:(self._registered_appearance_similarity(boxes[j],ci),float(scores[j])),reverse=True)
        return [(boxes[j],dict(grid_similarity=round(float(scores[j]),3))) for j in ranked[:6]]

    def _seed_generic_geometry(self,ci,box,proof,offset):
        """Cache correspondences measured in THIS worker frame, not a track."""
        # The colour gate may now use this fresh geometry for bounded exposure
        # compensation. A previous colour rejection in this frame is obsolete.
        getattr(self,'_localized_score_cache',{}).pop(tuple(box),None)
        x,y,r,b=box
        quad=np.asarray(proof['full_quad'],np.float32)+offset
        lo,hi=quad.min(axis=0),quad.max(axis=0)
        intersection=float(np.prod(np.maximum(0,np.minimum(hi,[r,b])-np.maximum(lo,[x,y]))))
        extent=intersection/max(1.,float(np.prod(hi-lo))+(r-x)*(b-y)-intersection)
        relative=(quad-[x,y])*[96./(r-x),96./(b-y)]
        key=(self.frame_idx,self._search_epoch)
        if getattr(self,'_pattern_cache_key',None)!=key:
            self._pattern_cache_key=key;self._pattern_cache={}
        lower,upper=relative.min(axis=0),relative.max(axis=0)
        self._pattern_cache[(ci,tuple(box))]=dict(similarity=-1.,
            geometry=dict(proof,extent_iou=round(extent,3)),full_quad=relative.tolist(),
            patch=tuple(lower)+tuple(upper-lower))
        if proof.get('observed_points') is not None:
            observed=np.float32(proof['observed_points'])+offset
            self._pattern_cache[(ci,tuple(box))]['motion_points']=((observed-[x,y])*[96./(r-x),96./(b-y)]).tolist()

    def _resume_generic_candidates(self,ci,queued,region,deadline):
        """Retain reference choices, remeasure geometry on current pixels."""
        left,top,right,bottom=region
        choices=queued.get('choices',{})
        resumed=[];projected=[];seen=set()
        for box in queued['boxes']:
            if not (left<=.5*(box[0]+box[2])<right and top<=.5*(box[1]+box[3])<bottom):continue
            choice=choices.get(tuple(box))
            resumed.append((box,dict(resumed=True,reference_choice=choice)))
            if choice is not None and choice[0]=='template':
                resumed[-1]=(box,dict(resumed=True,template_variant=tuple(choice[1:])))
                continue
            if choice is None or choice in seen or len(seen)>=2 or time.perf_counter()>=deadline:continue
            seen.add(choice);method,index=choice
            references=self.data.class_scene_refs(ci,method)
            if not 0<=index<len(references):continue
            x,y,r,b=box;w,h=r-x,b-y
            roi=(max(left,int(x-.3*w)),max(top,int(y-.3*h)),
                 min(right,int(r+.3*w)),min(bottom,int(b+.3*h)))
            sx,sy,sr,sb=roi
            if min(sr-sx,sb-sy)<24:continue
            found=project_reference_candidates(self.frame[sy:sb,sx:sr],
                [references[index]],method,time_budget=max(0.,deadline-time.perf_counter()))
            for candidate,proof in found:
                current=tuple(v+(sx if j%2==0 else sy) for j,v in enumerate(candidate))
                if _iou(current,box)<.25:continue
                proof=dict(proof,reference=index)
                # This proof was computed NOW. Queue entries never hold old
                # correspondences, transform matrices or acceptance scores.
                self._seed_generic_geometry(ci,current,proof,np.float32([sx,sy]))
                projected.append((current,dict(resumed=True,reference_choice=choice,
                    current_reference_geometry=True)))
        current_boxes={tuple(box) for box,_ in projected}
        return projected+[(box,proof) for box,proof in resumed if tuple(box) not in current_boxes]

    def _search_learned_parts(self,budget,selected_class=None,variants_left=2):
        """Experimental independent acquisition from saved local texture.

        One missing class gets a bounded supplemental job after ordinary search.
        A vote only proposes a crop; original coherent reference, classifier,
        veto, shape and display requirements remain mandatory.
        """
        from bbox_parts import part_vote_candidates
        started=time.perf_counter()
        deadline=min(started+budget if budget is not None else float('inf'),
                     getattr(self,'_generic_job_deadline',float('inf')))
        if selected_class is None:self._diag_generic_candidates={};self._diag_generic_source_work={}
        healthy={t['cls'] for t in self.scan_targets if t.get('verified') and t.get('pend',0)>=4 and t.get('fail',0)==0}
        eligible=getattr(self,'_part_eligible',set())
        classes=[ci for ci in range(len(self.data.labels)) if self._generic_class(ci) and ci not in healthy and ci in eligible]
        if not classes:return []
        if selected_class is None:
            cursor=getattr(self,'_part_class_cursor',0)
            ci=classes[cursor%len(classes)];self._part_class_cursor=cursor+1
        else:ci=selected_class
        references=self.data.class_spatial_refs(ci,patch_size=16)
        scales=sorted(set(float(s) for s in self.args.scan_scales if s>0),key=lambda s:abs(np.log(s)))
        upper=min((s for s in scales if s>1.),default=1.)
        near=[1.]+([float(np.sqrt(upper))] if upper>1 else [])
        rest=[s for s in scales if s not in near]
        poses=sorted(references,key=lambda r:(r['rotation'],r['reference'],r['flip']))
        variants=[(ref,factor) for factors in (near,rest) for ref in poses for factor in factors]
        states=getattr(self,'_part_variant_cursors',{});index=states.get(ci,0)%max(1,len(variants))
        if not variants or time.perf_counter()>=deadline:
            self._diag_generic_source_work[ci]=[dict(source='learned_parts',status='not_started_reference_budget')]
            return []
        ref,factor=variants[index];side=self.data.class_scan_fracs()[ci]*min(self.W,self.H)*factor
        ar=self.data.class_ar(ci);size=(side*min(1.,ar),side*min(1.,1./ar))
        if ref['rotation']%2:size=size[::-1]
        found=part_vote_candidates(self.frame,ref,size,max_candidates=1)
        self._part_variant_cursors=states
        work=dict(source='learned_parts',variant=index,variants_total=len(variants),reference=ref['reference'],
            flip=ref['flip'],rotation=ref['rotation'],factor=factor,candidates=len(found),checked=0)
        self._diag_generic_source_work.setdefault(ci,[]).append(work)
        self._diag_generic_candidates.setdefault(ci,[])
        incomplete=False
        for box,proof in found:
            if time.perf_counter()>=deadline:
                incomplete=True;break
            x,y,r,b=box
            pattern=match_fine_spatial_reference(self.frame[y:b,x:r],references,deadline=deadline)
            work['checked']+=1
            if pattern is IDENTITY_INCOMPLETE:
                incomplete=True;break
            reference=self._registered_appearance_similarity(box,ci)
            if not self._registered_identity_ok(reference,pattern,ci):
                if time.perf_counter()>=deadline:
                    incomplete=True;break
                # A proposal may also satisfy an established registration
                # path. Use its unchanged gates; never treat tile votes as
                # independent permission to admit an identity.
                pattern=self._registered_pattern(box,ci)
                if not self._registered_identity_ok(reference,pattern,ci):continue
            key=(self.frame_idx,self._search_epoch)
            if getattr(self,'_pattern_cache_key',None)!=key:self._pattern_cache_key=key;self._pattern_cache={}
            self._pattern_cache[(ci,box)]=pattern
            score=float(self._score_localized_boxes([box])[0,ci])
            accepted=bool(score>=self.args.scan_cos_th+.02 and self._new_white_identity_ok(box,ci,score)
                and self._shape_gate_ok(box,ci,False) and self._localized_geometry_ok(box,ci)
                and self._cs_gate_ok(box,cs_th=.16))
            self._diag_generic_candidates[ci].append(dict(box=box,score=score,proof=proof,accepted=accepted))
            if accepted:
                states[ci]=(index+1)%len(variants)
                work.update(elapsed_ms=(time.perf_counter()-started)*1000,completed=True)
                return [dict(cls=ci,box=box,occupancy=None,generic_boundary=True,proposal_source='learned_parts')]
        retries=getattr(self,'_part_variant_retries',{})
        retry_key=(ci,index)
        retry=bool(incomplete and retries.get(retry_key,0)<1)
        if retry:
            retries[retry_key]=1
        else:
            retries.pop(retry_key,None);states[ci]=(index+1)%len(variants)
        self._part_variant_retries=retries
        work.update(elapsed_ms=(time.perf_counter()-started)*1000,completed=not incomplete,
                    retry_variant=retry,status='incomplete' if incomplete else 'checked')
        if variants_left>1 and time.perf_counter()<deadline and not retry:
            return self._search_learned_parts(deadline-time.perf_counter(),selected_class=ci,variants_left=variants_left-1)
        return []

    def _search_generic_proposals(self,classes=None,region=None,budget=None):
        if classes is None and region is None and getattr(self,'_in_search_worker',False):
            turn=getattr(self,'_part_global_turn',0);self._part_global_turn=turn+1
            healthy={t['cls'] for t in self.scan_targets if t.get('verified') and t.get('pend',0)>=4 and t.get('fail',0)==0}
            pending=any(not t.get('ever_confirmed') and t.get('pend',0)<4 for t in self.scan_targets)
            if turn%4==3 and getattr(self,'_part_eligible',set())-healthy and not pending:
                return self._search_learned_parts(budget)
        proposals=[]
        self._diag_generic_candidates={}
        self._diag_generic_source_work={}
        search_deadline=min(time.perf_counter()+budget if budget is not None else float('inf'),
                            getattr(self,'_generic_job_deadline',float('inf')))
        deadline=search_deadline
        # Preserve the fast path, but reserve a final opportunity for strong
        # candidates whose indexed references could not establish identity.
        # The budget is soft: one OpenCV matching call cannot be interrupted.
        full_checks_left=getattr(self,'_generic_full_checks_left',2)
        reserve=min(.06,max(0.,budget*.25)) if budget is not None else 0.
        if not hasattr(self,'_generic_search_phases'):
            self._generic_search_phases={};self._generic_template_states={}
            self._generic_scene_states={}
        if not hasattr(self,'_generic_class_cursors'):self._generic_class_cursors={}
        if not hasattr(self,'_generic_pending_candidates'):self._generic_pending_candidates={}
        region=(0,0,self.W,self.H) if region is None else tuple(map(int,region))
        left,top,right,bottom=map(int,region)
        search_frame=self.frame[top:bottom,left:right]
        def translated(candidates):
            return [((b[0]+left,b[1]+top,b[2]+left,b[3]+top),proof) for b,proof in candidates]
        indices=list(range(len(self.data.labels)) if classes is None else classes)
        if classes is None and len(indices)>1 and getattr(self,'_in_search_worker',False):
            # Global acquisition must serve missing identities. Re-proposing a
            # currently healthy class both spends its slice and can terminate
            # the job before an unseen class runs; the foreground then ignores
            # that redundant proposal. Boundary refresh uses explicit local
            # jobs. A later loss makes the class eligible again next snapshot.
            healthy={t['cls'] for t in self.scan_targets if t.get('verified')
                     and t.get('pend',0)>=4 and t.get('fail',0)==0}
            missing=[ci for ci in indices if ci not in healthy]
            if healthy and missing:indices=missing
        if budget is not None and indices:
            scope=('global',) if classes is None else ('local',tuple(sorted(indices)))
            start=self._generic_class_cursors.get(scope,0)%len(indices)
            indices=indices[start:]+indices[:start]
            self._generic_class_cursors[scope]=(start+1)%len(indices)
        query_cache=getattr(self,'_generic_job_queries',{})
        pending_service=None
        if budget is not None and classes is None and getattr(self,'_in_search_worker',False):
            # A class-sized slice can expire during classification before a
            # queued ROI gets any identity check. Alternate one pending service
            # with ordinary round-robin acquisition; keep the job deadline and
            # original queue expiry. No old score or camera proof is reused.
            pending=[ci for ci in indices if ci in self._generic_pending_candidates
                     and self._generic_pending_candidates[ci]['expires']>time.perf_counter()
                     and self._generic_pending_candidates[ci].get('source') in
                         ('geometry','scale_geometry','template','initial_geometry')]
            if pending:
                service_turn=getattr(self,'_generic_pending_service_turn',True)
                self._generic_pending_service_turn=not service_turn
                if service_turn:
                    served=getattr(self,'_generic_pending_last_served',{})
                    pending_service=min(pending,key=lambda ci:served.get(ci,-1))
                    served[pending_service]=max(served.values(),default=-1)+1
                    self._generic_pending_last_served=served
                    indices=[pending_service]+[ci for ci in indices if ci!=pending_service]
        source_progress={}
        def scene_candidates(ci,method):
            source_progress['reference_cold']=(ci,method) not in self.data._scene_refs
            prepared=time.perf_counter()
            references=self.data.class_scene_refs(ci,method)
            source_progress.update(reference_prepare_ms=round((time.perf_counter()-prepared)*1000,3),
                learned_references=len(references))
            remaining=deadline-time.perf_counter()
            if remaining<=0:
                source_progress.update(completed=False,checked=0,not_started='reference_prepare_budget')
                return []
            return translated(project_reference_candidates(search_frame,references,method,
                time_budget=remaining if budget is not None else None,
                state=self._generic_scene_states.setdefault((ci,method),{}) if budget is not None else None,
                expand_empty=acquiring,progress=source_progress,
                query_cache=query_cache.setdefault((region,method),{})))
        for position,ci in enumerate(indices):
            now=time.perf_counter()
            if now>=search_deadline:
                for unserved in indices[position:]:
                    self._diag_generic_source_work[unserved]=[dict(status='not_started_job_budget',
                        phase=self._generic_search_phases.get(unserved,0))]
                break
            deadline=search_deadline if budget is None else min(search_deadline,
                now+(search_deadline-now)/max(1,len(indices)-position))
            full_deadline=deadline
            if ci==pending_service:deadline=search_deadline
            if not self._generic_class(ci):continue
            self._diag_generic_candidates[ci]=[]
            self._diag_generic_source_work[ci]=[]
            acquiring=not any(
                t['cls']==ci and t.get('ever_confirmed') for t in self.scan_targets)
            fraction=self.data.class_scan_fracs()[ci]*min(self.W,self.H)/max(1,min(search_frame.shape[:2]))
            phase_one=((1,'scale_geometry',lambda:scene_candidates(ci,'sift')),
                       (1,'object_region',lambda:self._generic_region_candidates(ci,region)))
            # Expensive unverified regions must not consume the first SIFT
            # opportunity for a missing identity. Preserve the original order
            # and reference work after confirmation. Pending local searches
            # still need their first independent boundary confirmation.
            if not acquiring:phase_one=phase_one[::-1]
            sources=(
                (0,'geometry',lambda:scene_candidates(ci,'orb')),
                (0,'bright_component',lambda:self._generic_bright_candidates(ci,region)),
                (0,'colour_component',lambda:translated(colour_component_candidates(search_frame,self.data.class_hist(ci)))
                    if chromatic_model(self.data.class_hist(ci)) is not None else []),
                *phase_one,
                (2,'template',lambda:translated(template_reference_candidates(search_frame,self.data.class_reference_views(ci),
                    fraction,self.args.scan_scales,progress=source_progress,
                    state=self._generic_template_states.setdefault(ci,{}) if budget is not None else None,
                    time_budget=min(.12,max(.001,deadline-time.perf_counter())) if budget is not None else None,
                    appearance_ok=lambda box:self._registered_appearance_similarity(
                        (box[0]+left,box[1]+top,box[2]+left,box[3]+top),ci)>=.70))),
                (3,'grid',lambda:(self._generic_texture_candidates(ci,region,deadline) if acquiring else [])
                    or self._generic_grid_candidates(ci,region)),
                (4,'refined',lambda:self._generic_expanded_candidates(ci,region)))
            phase=self._generic_search_phases.get(ci,0)
            queued=self._generic_pending_candidates.pop(ci,None)
            if queued is not None and queued['expires']<=time.perf_counter():queued=None
            candidate_choices={} if queued is None else dict(queued.get('choices',{}))
            # Rotate sources WITHIN a phase too. Otherwise the first ORB
            # pass can consume every phase-0 budget before components run.
            active=[entry for entry in sources if entry[0]==phase]
            if budget is not None and len(active)>1:
                turns=getattr(self,'_generic_source_turns',{})
                turn=turns.get((ci,phase),0)%len(active)
                sources=tuple(active[turn:]+active[:turn])
                turns[(ci,phase)]=turn+1
                self._generic_source_turns=turns
            if queued is not None:
                sources=((phase,'resume',lambda:self._resume_generic_candidates(ci,queued,region,deadline)),)+tuple(sources)
                if ci==pending_service:sources=sources[:1]
            preferred=getattr(self,'_generic_template_preferences',{}).get(ci)
            if (classes is None and ci!=pending_service and preferred is not None
                    and getattr(self.args,'confirmed_template_reacquire',False)):
                # One learned variant selected by a previously confirmed
                # foreground target. Fresh pixels and all original gates are
                # required; the ordinary exhaustive cursor is not advanced.
                sources=((phase,'preferred_template',lambda:translated(template_reference_candidates(
                    search_frame,self.data.class_reference_views(ci),fraction,self.args.scan_scales,
                    max_candidates=3,only_variant=preferred,
                    appearance_ok=lambda box:self._registered_appearance_similarity(
                        (box[0]+left,box[1]+top,box[2]+left,box[3]+top),ci)>=.70))),)+tuple(sources)
            accepted=False
            deferred=False
            full_candidates={}
            def defer_boxes(boxes,source):
                nonlocal deferred
                if not boxes:return
                expires=queued['expires'] if source=='resume' and queued is not None else time.perf_counter()+.35
                # Sources already bound their candidate batches. Keep the
                # remaining coordinates/learned choices until the same expiry;
                # a second prefix limit can discard the only provable ROI.
                # No camera proof or acceptance score is queued.
                selected=list(dict.fromkeys(tuple(b) for b in boxes))
                self._generic_pending_candidates[ci]=dict(expires=expires,boxes=selected,
                    choices={b:candidate_choices[b] for b in selected if b in candidate_choices},
                    source=queued.get('source') if source=='resume' and queued is not None else source)
                deferred=True
            def fast_expired():
                return time.perf_counter()>=(deadline-reserve if full_candidates else deadline)
            def measured_candidate(box,score,diagnostic):
                if time.perf_counter()>=deadline:return box,score
                fitted=self._measured_generic_extent(box,self._registered_pattern(box,ci))
                if fitted is None:return box,score
                self._cache_measured_geometry(box,fitted[0],fitted[1],ci,self._registered_pattern(box,ci))
                aligned=tuple(fitted[0]);current=float(self._score_localized_boxes([aligned])[0,ci])
                valid=(current>=self.args.scan_cos_th+.02 and self._new_white_identity_ok(aligned,ci,current)) \
                    or self._generic_geometry_admission(aligned,ci,current)
                diagnostic.update(measured_box=aligned,measured_score=round(current,3),measured_accepted=bool(valid))
                return (aligned,current) if valid else (box,score)
            for group,source,find in sources:
                if budget is not None and group!=phase:continue
                if budget is not None and fast_expired():
                    self._diag_generic_source_work[ci].append(dict(source=source,phase=phase,
                        status='not_started_source_budget'))
                    continue
                source_progress={};source_started=time.perf_counter()
                candidates=find()
                self._diag_generic_source_work[ci].append(dict(source=source,phase=phase,
                    elapsed_ms=round((time.perf_counter()-source_started)*1000,3),
                    remaining_ms=round((deadline-time.perf_counter())*1000,3),
                    candidates=len(candidates),progress=dict(source_progress)))
                if not candidates:continue
                boxes=[box for box,proof in candidates]
                if source in ('geometry','scale_geometry'):
                    for box,proof in candidates:self._seed_generic_geometry(ci,box,proof,np.float32([left,top]))
                for box,proof in candidates:
                    method=proof.get('detector');index=proof.get('reference')
                    if method in ('orb','sift') and isinstance(index,(int,np.integer)):
                        candidate_choices[tuple(box)]=(method,int(index))
                    elif proof.get('template_variant') is not None:
                        candidate_choices[tuple(box)]=('template',)+tuple(proof['template_variant'])
                    elif proof.get('reference_choice') is not None:
                        candidate_choices[tuple(box)]=tuple(proof['reference_choice'])
                scores=self._score_localized_boxes(boxes)[:,ci]
                order=(list(range(len(boxes))) if source=='refined' else
                    sorted(range(len(boxes)),key=lambda j:(self._registered_appearance_similarity(boxes[j],ci),float(scores[j])),reverse=True))
                for order_position,j in enumerate(order):
                    box,proof=candidates[j];score=float(scores[j])
                    # A negative score is an existing hard colour/background/
                    # negative-example veto on these pixels. Neither admission
                    # route can accept it; do not spend a later queue turn on
                    # this already rejected source proposal. Fresh searches
                    # can generate the coordinates again on a changed frame.
                    if score<0.:
                        self._diag_generic_candidates[ci].append(dict(box=box,score=round(score,3),
                            source=source,proof=proof,accepted=False,hard_veto=True))
                        continue
                    cached=(self._pattern_cache.get((ci,tuple(box)))
                        if getattr(self,'_pattern_cache_key',None)==(self.frame_idx,self._search_epoch) else None)
                    complete_current=(score>=self.args.scan_cos_th+.02 and cached is not None
                        and self._registered_identity_ok(self._registered_appearance_similarity(box,ci),cached,ci))
                    # Scoring can use up a soft budget AFTER both geometry and
                    # feature gates have finished. Deliver that completed
                    # current-frame proof; do not defer it merely to redo it.
                    # No extra matching or inference is allowed by this path.
                    if fast_expired() and not complete_current:
                        defer_boxes([boxes[k] for k in order[order_position:] if scores[k]>=0.],source)
                        break
                    valid=complete_current or (score>=self.args.scan_cos_th+.02 and self._new_white_identity_ok(box,ci,score)) \
                          or self._generic_geometry_admission(box,ci,score)
                    diagnostic=dict(box=box,score=round(score,3),source=source,proof=proof,
                                    accepted=bool(valid))
                    self._diag_generic_candidates[ci].append(diagnostic)
                    if valid:
                        box,score=measured_candidate(box,score,diagnostic)
                        proposals.append(dict(cls=ci,box=box,occupancy=None,generic_boundary=True,
                                              proposal_source=source))
                        accepted=True
                        break
                    if full_checks_left and getattr(self,'_generic_identity_fast',False) and score>=0.:
                        reference=self._registered_appearance_similarity(box,ci)
                        if reference>=.58:
                            key=tuple(box)
                            rank=(score>=self.args.scan_cos_th+.02,reference,score)
                            full_candidates[key]=(rank,box,score,source,diagnostic)
                            if len(full_candidates)>2:
                                worst=min(full_candidates,key=lambda k:full_candidates[k][0])
                                del full_candidates[worst]
                if accepted:break
            if not accepted:
                complete_order=sorted(full_candidates.values(),key=lambda v:v[0],reverse=True)
                for complete_position,(_,box,score,source,diagnostic) in enumerate(complete_order):
                    # The priority turn pays for a current quick identity
                    # attempt. Keep the original class slice for starting a
                    # second, full matcher so a slow failed quick check cannot
                    # immediately trigger another indivisible matching call.
                    if not full_checks_left or time.perf_counter()>=full_deadline:
                        if not deferred:
                            remaining=[]
                            for item in complete_order[complete_position:]:
                                original=item[1]
                                fitted=self._cached_generic_reference_fit(original,ci)
                                if fitted is not None:remaining.append(tuple(fitted[0]))
                                remaining.append(original)
                            # Only coordinates enter the bounded queue. The
                            # next source frame must establish identity again.
                            defer_boxes(remaining,source)
                        break
                    full_checks_left-=1
                    if hasattr(self,'_generic_full_checks_left'):
                        self._generic_full_checks_left=full_checks_left
                    previous=self._generic_identity_fast
                    try:
                        # A failed quick crop can already contain a distributed
                        # current geometric match. Re-crop its learned extent
                        # before discarding it and repeating full matching.
                        fitted=self._cached_generic_reference_fit(box,ci)
                        valid=False
                        if fitted is not None:
                            aligned=tuple(fitted[0])
                            aligned_score=float(self._score_localized_boxes([aligned])[0,ci])
                            aligned_valid=(aligned_score>=self.args.scan_cos_th+.02
                                and self._new_white_identity_ok(aligned,ci,aligned_score)) \
                                or self._generic_geometry_admission(aligned,ci,aligned_score)
                            diagnostic.update(cached_geometry_box=aligned,
                                cached_geometry_score=round(aligned_score,3),
                                cached_geometry_accepted=bool(aligned_valid))
                            if aligned_valid:
                                box,score,valid=aligned,aligned_score,True
                        self._generic_identity_fast=False
                        # A cached None has no _fast_only marker. Explicitly
                        # invalidate this failed quick result before upgrading.
                        if not valid and time.perf_counter()<deadline:
                            self._pattern_cache.pop((ci,tuple(box)),None)
                            valid=(score>=self.args.scan_cos_th+.02 and self._new_white_identity_ok(box,ci,score)) \
                                  or self._generic_geometry_admission(box,ci,score)
                        if not valid and time.perf_counter()<deadline:
                            # Components describe visible material, which need
                            # not have the same extent as the learned ROI.
                            # Repair from current reference correspondences in
                            # the worker, then repeat ALL feature/identity gates.
                            fitted=self._fit_generic_reference(box,ci)
                            if fitted is not None:
                                aligned=tuple(fitted[0])
                                aligned_score=float(self._score_localized_boxes([aligned])[0,ci])
                                aligned_valid=(aligned_score>=self.args.scan_cos_th+.02
                                    and self._new_white_identity_ok(aligned,ci,aligned_score)) \
                                    or self._generic_geometry_admission(aligned,ci,aligned_score)
                                diagnostic.update(aligned_box=aligned,
                                    aligned_score=round(aligned_score,3),aligned_accepted=bool(aligned_valid))
                                if aligned_valid:
                                    box,score,valid=aligned,aligned_score,True
                    finally:
                        self._generic_identity_fast=previous
                    diagnostic.update(full_verification=True,accepted=bool(valid))
                    if valid:
                        box,score=measured_candidate(box,score,diagnostic)
                        proposals.append(dict(cls=ci,box=box,occupancy=None,generic_boundary=True,
                                              proposal_source=source,full_verification=True))
                        accepted=True
                        break
            if accepted:
                self._generic_pending_candidates.pop(ci,None)
                # A worker proposal is not yet a foreground confirmation.
                # Give its learned geometry one fresh local re-projection,
                # rather than restarting a full source cycle if transport or
                # the next frame rejects the first box. Resume results cannot
                # requeue themselves; expiry and every current gate stay intact.
                if acquiring and proposals[-1].get('proposal_source')=='scale_geometry':
                    proof=next((entry.get('proof',{}) for entry in self._diag_generic_candidates[ci]
                        if entry.get('accepted') and entry.get('proof',{}).get('expanded_reference')),None)
                    if proof is not None and isinstance(proof.get('reference'),(int,np.integer)):
                        current_box=tuple(proposals[-1]['box'])
                        self._generic_pending_candidates[ci]=dict(expires=time.perf_counter()+.35,
                            boxes=[current_box],choices={current_box:(proof['detector'],int(proof['reference']))},
                            source='initial_geometry')
                chosen=next((entry.get('proof',{}).get('template_variant')
                             for entry in self._diag_generic_candidates[ci]
                             if entry.get('accepted') and entry.get('proof',{}).get('template_variant') is not None),None)
                if chosen is not None:proposals[-1]['template_variant']=tuple(chosen)
            # Deferred candidates have their own bounded queue. They must not
            # pin the source phase and starve fresh geometry/grid proposals.
            # A resume-only turn did not visit its fresh source phase.
            visited=any(item.get('source') not in ('resume','preferred_template')
                        and 'elapsed_ms' in item for item in self._diag_generic_source_work[ci])
            self._generic_search_phases[ci]=0 if accepted else (phase+1)%5 if visited else phase
            if classes is None and not accepted and phase==4 and any(
                    v.get('source') in ('grid','refined') and 'elapsed_ms' in v
                    for v in self._diag_generic_source_work[ci]):
                eligible=getattr(self,'_part_eligible',set());eligible.add(ci);self._part_eligible=eligible
            if accepted and getattr(self,'_in_search_worker',False):
                # Deliver the first proved target now. Other classes rotate
                # into the next fresh job rather than aging this result.
                break
        return proposals

    def _search_colour_proposals(self,exclude_classes=()):
        proposals=[]
        self._diag_colour_candidates={}
        for ci,white in enumerate(self._bright_localizer_flags()):
            if white or ci in exclude_classes:continue
            fits=colour_component_candidates(self.frame,self.data.class_hist(ci))
            if not fits:continue
            boxes=[b for b,e in fits];scores=self._score_localized_boxes(boxes)[:,ci]
            evidence_rows=copy.deepcopy(self._last_localized_evidence)
            self._diag_colour_candidates[ci]=[]
            passed=[]
            for j in np.argsort(scores)[::-1]:
                box,evidence=fits[j];score=float(scores[j])
                reference=self._registered_appearance_similarity(box,ci)
                pattern=self._registered_pattern(box,ci) if score>=self.args.scan_cos_th+.02 else None
                reason=(f'分类或视图门不足({max(0.,score):.0%}<{self.args.scan_cos_th+.02:.0%})' if score<self.args.scan_cos_th+.02 else
                        '学习身份不符' if not self._new_white_identity_ok(box,ci,score) else
                        '内部形状不足' if not self._shape_gate_ok(box,ci,False) else
                        '与周围太像' if not self._cs_gate_ok(box,cs_th=.16) else
                        '肤色过多' if not self._generic_class(ci) and self._skin_ratio(box)>.65 else '通过')
                self._diag_colour_candidates[ci].append(dict(box=box,score=round(score,3),
                    reference=round(reference,3),pattern=pattern,reason=reason,
                    component=evidence,evidence=evidence_rows[j][ci]))
                if reason!='通过':
                    continue
                passed.append(j)
            if passed:
                pick=passed[0]
                anchor=boxes[pick];area=max(1,(anchor[2]-anchor[0])*(anchor[3]-anchor[1]))
                enclosing=[]
                for j in passed[1:]:
                    box=boxes[j];ratio=((box[2]-box[0])*(box[3]-box[1]))/area
                    intersection=max(0,min(box[2],anchor[2])-max(box[0],anchor[0])) \
                        *max(0,min(box[3],anchor[3])-max(box[1],anchor[1]))
                    if fits[j][1].get('primary_box')==fits[pick][1].get('primary_box') \
                            and 1.1<=ratio<=1.6 and intersection>=.95*area:
                        enclosing.append(j)
                if enclosing:pick=max(enclosing,key=lambda j:(boxes[j][2]-boxes[j][0])*(boxes[j][3]-boxes[j][1]))
                box,evidence=fits[pick]
                proposals.append(dict(cls=ci,box=box,occupancy=evidence['occupancy'],colour_boundary=True))
        return proposals

    def _accept_localized_target(self, tgt, box, score, source, boundary=True,
                                 motion=None, occupancy=None):
        if not self._bright_localizer_flags()[tgt['cls']]:
            return self._accept_colour_target(tgt,box,score,source,boundary=boundary)
        # Temporal appearance and LK also remain stable on a wrongly acquired
        # background. Recheck against LEARNING references, never the track bank.
        has_refs=bool(self.data.class_visual_refs(tgt["cls"]))
        now = time.perf_counter()
        tgt['admission_pending']=False
        tgt['identity_blur_bridge'] = False
        pattern = self._registered_pattern(box, tgt['cls']) if has_refs else None
        if pattern is not None:
            tgt['registered_pattern_similarity'] = round(pattern['similarity'], 3)
            tgt['pattern_evidence_frame'] = self.frame_idx
        # Recheck learned spatial structure on EACH accepted frame. A scene can
        # remain self-similar after the object leaves; its recent track bank is
        # not sufficient evidence for keeping a formal identity.
        pattern_failed = pattern is not None and not self._registered_identity_ok(
            float(tgt.get('registered_identity_similarity', 0.)), pattern, tgt['cls'])
        # A matching track/template is not a fresh segmentation. Periodically
        # require a new independent boundary, including on static backgrounds.
        segmentation_frame = int(tgt.get("segmentation_frame",
                                  self.frame_idx-int(tgt.get("boundary_age", 0))))
        segmentation_expired = bool(tgt.get("ever_confirmed") and not boundary and (
            self.frame_idx-segmentation_frame >= 90
            or now-tgt.get("segmentation_time", now) >= 3.))
        identity_due=(segmentation_expired or pattern_failed or (has_refs and (
            boundary or tgt.get("identity_blocked",False) or tgt.get('boundary_pending',False)
            or self.frame_idx-int(tgt.get("identity_check_frame",-12))>=12
            or (not tgt.get("ever_confirmed") and tgt.get("pend",0)>=3))))
        if identity_due:
            reference=self._registered_appearance_similarity(box,tgt["cls"]) if has_refs else 1.
            identity_ok = self._registered_identity_ok(reference, pattern, tgt['cls'])
            learned_identity_ok = identity_ok
            # A blur bridge has a fixed deadline from an independently learned
            # identity. Bridged frames never renew it, nor can CSRT alone use it.
            if not identity_ok and tgt.get('ever_confirmed') and pattern is not None \
                    and motion is not None and tuple(box)==tuple(motion.box) \
                    and motion.inlier_ratio>=.80 \
                    and motion.coverage[0]>=.20 and motion.coverage[1]>=.25 \
                    and 0.<=now-tgt.get('learned_identity_time',0.)<=.25 \
                    and self.frame_idx-int(tgt.get('learned_identity_frame',-1000))<=8 \
                    and score>=self.args.scan_cos_th+.02 \
                    and reference>=.50 and pattern['similarity']>=.68:
                record=getattr(self,'_diag_localized',{}).get(tgt['cls'],{})
                current_record=(record.get('frame')==self.frame_idx
                                and tuple(record.get('box',()))==tuple(box))
                current_views=all(sum(float(v)>=self.args.scan_cos_th+.02
                                      for v in record.get(name,[]))>=2
                                  for name in ('color','background','negative'))
                sharp=reference_sharpness(cv2.cvtColor(self.frame,cv2.COLOR_BGR2GRAY),box)
                blurred=sharp<.65*float(tgt.get('learned_identity_sharpness',0.))
                if current_record and current_views and blurred:
                    identity_ok=True
                    tgt['identity_blur_bridge']=True
            # A fresh correlation proposal supplies position continuity; the
            # learned CNN's multiple views supply CURRENT identity when the
            # single saved thumbnail is marginal at distance. Every veto still
            # has to pass, and a new/global target cannot use this condition.
            if not identity_ok and tgt.get('ever_confirmed') and pattern is not None \
                    and tgt.get('_correlation_frame')==self.frame_idx \
                    and tgt.get('_correlation_box')==tuple(box) \
                    and score>=self.args.scan_cos_th+.08 \
                    and reference>=.50 and pattern['similarity']>=.68:
                record=getattr(self,'_diag_localized',{}).get(tgt['cls'],{})
                identity_ok=all(sum(float(v)>=0 for v in record.get(name,[]))>=2
                                for name in ('color','background','negative'))
            tgt['registered_pattern_similarity'] = None if pattern is None else round(pattern['similarity'], 3)
            tgt['identity_evidence_frame'] = self.frame_idx
            geometry=self._localized_geometry_ok(box,tgt["cls"]) \
                     and self._shape_gate_ok(box,tgt["cls"],True)
            support,complete=endpoint_boundary_support(self.frame,box)
            tgt["endpoint_support"]=tuple(round(v,3) for v in support)
            tgt['boundary_source']='endpoints'
            if not complete and learned_identity_ok and pattern is not None \
                    and pattern['similarity']>=.78 and score>=self.args.scan_cos_th+.02 \
                    and (measured_tapered_boundary(self.frame,box,self.data.class_ar(tgt['cls']))
                         or measured_narrow_tip(self.frame,box,self.data.class_ar(tgt['cls']))):
                complete=True
                tgt['boundary_source']='measured_taper'
            # Only an already confirmed identity with recent real endpoint
            # evidence can bridge a short blur using reliable geometric motion.
            bridge=bool(tgt.get("ever_confirmed") and motion is not None
                and motion.inlier_ratio>=.75 and motion.coverage[0]>=.2 and motion.coverage[1]>=.25
                and self.frame_idx-int(tgt.get("complete_boundary_frame",-1000))<=18
                and time.perf_counter()-tgt.get("complete_boundary_time",0.)<=.6)
            boundary_ok=complete or bridge
            body_distance, body_fraction, body_ratio = self._registered_body_shape(box, tgt["cls"])
            tgt["body_shape_distance"] = (round(body_distance, 3)
                if body_distance is not None and np.isfinite(body_distance) else None)
            tgt["body_fraction"] = round(body_fraction, 3)
            tgt["body_elongation_ratio"] = round(body_ratio, 3) if np.isfinite(body_ratio) else None
            # Blur and glare split the U-shaped white shell into thin outlines.
            # Hu moments become unstable on these contours; their oriented span
            # remains useful. Both measurements concern ONE connected component.
            body_complete = body_distance is None or body_distance <= .9 or body_ratio <= 2.
            body_bridge = bool(bridge and
                self.frame_idx-int(tgt.get("complete_body_frame", -1000)) <= 18 and
                now-tgt.get("complete_body_time", 0.) <= .6)
            body_ok = body_complete or body_bridge
            tgt["registered_identity_similarity"]=reference
            if not identity_ok or not geometry or not boundary_ok or not body_ok or segmentation_expired:
                tgt["verified"],tgt["coasting"],tgt["identity_blocked"]=False,False,True
                # A current learned identity can remain a HIDDEN boundary
                # candidate. Do not delete its position merely because a thin
                # tip is unclear. This fixed deadline never renews identity,
                # confirmation, or the independent segmentation lease.
                record=getattr(self,'_diag_localized',{}).get(tgt['cls'],{})
                pending_new=bool(not tgt.get('ever_confirmed') and has_refs
                    and learned_identity_ok and pattern is not None and pattern['similarity']>=.82
                    and max(support)>=.30 and record.get('frame')==self.frame_idx
                    and tuple(record.get('box',()))==tuple(box)
                    and all(sum(float(v)>=self.args.scan_cos_th+.02 for v in record.get(name,[]))>=2
                            for name in ('color','background','negative')))
                boundary_pending=bool((tgt.get('ever_confirmed') or pending_new) and has_refs
                    and identity_ok and geometry and body_ok and not boundary_ok
                    and not segmentation_expired and score>=self.args.scan_cos_th+.02)
                if boundary_pending:
                    tgt.setdefault('boundary_pending_since',now)
                    boundary_pending=0.<=now-tgt['boundary_pending_since']<=1.
                tgt['boundary_pending']=boundary_pending
                if boundary_pending:
                    tgt['identity_blocked']=False
                    tgt['box'],tgt['ema']=tuple(box),None
                    tgt['asize']=(box[2]-box[0],box[3]-box[1])
                    if pending_new:
                        tgt['admission_pending']=True
                        tgt['cos']=tgt['gcos']=score
                        tgt.setdefault('pend',0)
                        tgt.setdefault('last_verified_time',0.)
                        if occupancy is not None:tgt['boundary_occupancy']=occupancy
                    self._init_target_motion(tgt)
                    if motion is not None:
                        tgt['velocity']=motion.velocity
                reason=(f"独立身份复核失败·注册外观{reference:.0%}、内部结构"
                    + (f"{pattern['similarity']:.0%}" if pattern is not None else "无参考") if not identity_ok else
                    "独立身份复核失败·完整框形状不符" if not geometry else
                    f"独立身份复核失败·端部边界不足({support[0]:.0%}/{support[1]:.0%})" if not boundary_ok else
                    "独立身份复核失败·连续亮色主体轮廓与学习图不符" if not body_ok else
                    "独立身份复核失败·过久未获得新分割边界")
                tgt["pending_reason"]=reason
                if boundary_pending:
                    reason='身份通过·等待端部重新定位'
                    tgt['pending_reason']=reason
                self._diag_reject[tgt["cls"]]=reason
                # Search and tracking may check the same frame twice. Failed
                # independent checks count once and never refresh the lease.
                if tgt.get("identity_fail_frame")!=self.frame_idx:
                    tgt["fail"]=int(tgt.get("fail",0))+1
                    tgt["identity_fail_frame"]=self.frame_idx
                if tgt["fail"]>6 and not boundary_pending and tgt in self.scan_targets:
                    self.scan_targets.remove(tgt)
                return False
            tgt["identity_check_frame"]=self.frame_idx
            tgt["identity_blocked"]=False
            tgt['boundary_pending']=False
            tgt.pop('boundary_pending_since',None)
            if learned_identity_ok and complete and body_complete:
                tgt['learned_identity_time']=now
                tgt['learned_identity_frame']=self.frame_idx
                tgt['learned_identity_sharpness']=reference_sharpness(
                    cv2.cvtColor(self.frame,cv2.COLOR_BGR2GRAY),box)
            if body_complete:
                tgt["complete_body_frame"] = self.frame_idx
                tgt["complete_body_time"] = now
            if complete:
                tgt["complete_boundary_frame"]=self.frame_idx
                tgt["complete_boundary_time"]=time.perf_counter()
        # Global search and tracking can both run in one frame; count it once.
        if boundary:
            tgt["boundary_votes"]=int(tgt.get("boundary_votes",0))+1
            tgt["segmentation_frame"] = self.frame_idx
            tgt["segmentation_time"] = now
        if tgt.get("last_evidence_frame") != self.frame_idx:
            limit=3 if self._async_search is not None and not tgt.get("ever_confirmed") \
                       and tgt.get("boundary_votes",0)<2 else 4
            tgt["pend"] = min(limit, int(tgt.get("pend", 0)) + 1)
        tgt["last_evidence_frame"] = self.frame_idx
        tgt["ever_confirmed"] = bool(tgt.get("ever_confirmed") or tgt["pend"] >= 4)
        if boundary:
            tgt["ema"], tgt["box"] = self._ema_box(box, tgt.get("ema"))
        else:
            tgt["box"] = box
            tgt["ema"] = None
        tgt["asize"] = (box[2]-box[0], box[3]-box[1])
        tgt["cos"] = tgt["gcos"] = score
        tgt["verified"], tgt["coasting"] = True, False
        tgt["last_verified_time"] = time.perf_counter()
        if pattern is not None and pattern['similarity'] >= .78 and score >= self.args.scan_cos_th+.02:
            tgt['strong_identity_time'] = now
        tgt["fail"] = 0
        tgt["pending_reason"] = ""
        tgt["boundary_age"] = 0 if boundary else int(tgt.get("boundary_age", 0)) + 1
        if occupancy is not None:
            tgt["boundary_occupancy"] = occupancy
        if motion is not None:
            tgt["velocity"] = motion.velocity
        if motion is not None and not boundary:
            tgt["motion_gray"] = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            tgt["motion_points"] = motion.points
            tgt["motion_ref_age"] = 0
            tgt['motion_capture_time'] = getattr(self, '_capture_time', None)
            tgt["motion_corners"] = motion.corners
            if len(motion.points) < 20 and (score >= self.args.scan_cos_th + .02 or (
                    motion.inlier_ratio>=.85 and motion.coverage[0]>=.2 and motion.coverage[1]>=.25)):
                corners=tgt.get("motion_corners")
                self._init_target_motion(tgt)
                tgt["motion_corners"]=corners
        else:
            self._init_target_motion(tgt)
        # Keep the template separate from the LK reference. LK must advance
        # each frame, while the recovery image must not become a blurred crop.
        sharp=reference_sharpness(tgt["motion_gray"],tgt["box"])
        previous_sharp=float(tgt.get("recovery_sharpness",0.))
        if score>=self.args.scan_cos_th+.02 and sharp>=max(8.,.65*previous_sharp):
            tgt["recovery_gray"]=tgt["motion_gray"]
            tgt["recovery_box"]=tgt["box"]
            tgt["recovery_time"]=time.perf_counter()
            tgt["recovery_sharpness"]=sharp
        if tgt["ever_confirmed"]:
            descriptor=describe_appearance(self.frame,tgt["box"])
            if descriptor is not None:
                bank=tgt.setdefault("appearance_bank",[])
                match=max((float(descriptor@v) for v in bank),default=1.)
                trusted_update = score>=self.args.scan_cos_th+.02 or (
                    motion is not None and motion.inlier_ratio>=.85
                    and motion.coverage[0]>=.2 and motion.coverage[1]>=.25 and match>=.85)
                if trusted_update and (not bank or .8<=match<.995):
                    bank.append(descriptor)
                    if len(bank)>6:
                        del bank[1]
        self._diag_reject[tgt["cls"]] = f"{source}{score:.0%}"
        return True

    def _track_localized_white(self, tgt, keep_th):
        """Motion carries an existing identity; new identities need the lock line."""
        if not hasattr(self, "_diag_reject"):
            self._diag_reject = {}
        ci, b = tgt["cls"], tgt["box"]
        confirmed = bool(tgt.get("ever_confirmed") or tgt["pend"] >= 4)
        if confirmed:
            tgt.setdefault("last_verified_time",time.perf_counter())
        correlation_box=None
        if confirmed and self._async_search is not None and min(b[2]-b[0],b[3]-b[1])<=160:
            tracker=tgt.get('_position_tracker')
            try:
                if tracker is None and tgt.get('fail',0)==0:
                    tracker=LocalCorrelationTracker(self.frame,b)
                    tgt['_position_tracker']=tracker
                    correlation_box=b
                elif tracker is not None:
                    correlation_box=tracker.update(self.frame)
                    if correlation_box is None:
                        tgt.pop('_position_tracker',None)
            except (cv2.error,AttributeError):
                tgt.pop('_position_tracker',None)
        candidate_wait=min(1.5,max(.65,.25+2.*self._search_last_ms*.001))
        if self._async_search is not None and not confirmed \
                and time.perf_counter()-tgt.get("candidate_started",time.perf_counter())>candidate_wait:
            self.scan_targets.remove(tgt)
            self._diag_reject[ci]="候选未通过独立边界复核"
            return
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        motion = track_points(tgt.get("motion_gray"), gray, tgt.get("motion_points"), b,
                              tgt.get("motion_corners"))
        tgt["motion_quality"] = 0. if motion is None else motion.inlier_ratio
        tgt['motion_coverage'] = None if motion is None else tuple(round(v,3) for v in motion.coverage)
        capture_time = getattr(self, '_capture_time', None)
        reference_time = tgt.get('motion_capture_time')
        period = 1. / max(5., float(getattr(self, '_perf_current', {}).get('camera_fps', 30.)))
        elapsed_steps = ((capture_time-reference_time)/period
            if capture_time is not None and reference_time is not None else None)
        if motion is not None:
            steps = max(1., elapsed_steps) if elapsed_steps is not None else int(tgt.get('motion_ref_age',0))+1
            motion = motion._replace(velocity=tuple(v/steps for v in motion.velocity))
        if motion is not None:
            predicted = motion.box
        else:
            steps = min(8., max(1., elapsed_steps)) if elapsed_steps is not None else min(3, int(tgt.get("fail", 0))+1)
            velocity = tuple(v*steps for v in tgt.get("velocity", (0.,0.)))
            predicted = shift_box(b, velocity, self.W, self.H)
        template_relocation=None
        if motion is None and tgt.get("motion_ref_age",0)<=3:
            template_relocation=relocate_template(tgt.get("motion_gray"),gray,b,min_similarity=.90)
            if template_relocation is not None:
                predicted=template_relocation[0]
        trusted_identity = time.perf_counter()-tgt.get('strong_identity_time', 0.) <= .6
        if motion is None and template_relocation is None and confirmed \
                and time.perf_counter()-tgt.get("recovery_time",0.) <= (.45 if trusted_identity else .25):
            search=recovery_region(b,tgt.get("velocity",(0.,0.)),self.W,self.H,tgt.get("fail",0))
            template_relocation=relocate_template(tgt.get("recovery_gray"),gray,
                tgt.get("recovery_box",b),min_similarity=.85,search_box=search)
            if template_relocation is not None:
                predicted=template_relocation[0]
        tgt["verified"], tgt["coasting"] = False, False
        lock_th = self.args.scan_cos_th + .02
        required = keep_th if confirmed and (motion is not None or template_relocation is not None) else lock_th
        age = int(tgt.get("boundary_age", 0))
        strong_motion = motion is not None and motion.inlier_ratio >= .75 \
                        and motion.coverage[0] >= .2 and motion.coverage[1] >= .25
        refresh_due = self.frame_idx-int(tgt.get("last_boundary_attempt", -6)) >= 6
        score_cache = {}

        def score_box(box):
            box = tuple(box)
            if box in score_cache:
                score, reason, evidence = score_cache[box]
                self._diag_localized[ci] = evidence
                return score, reason
            score = float(self._score_localized_boxes([box])[0, ci])
            evidence = self._last_localized_evidence[0][ci]
            self._diag_localized[ci] = evidence
            descriptor=describe_appearance(self.frame,box)
            bank=tgt.get("appearance_bank",[])
            similarity=max((float(descriptor@v) for v in bank),default=0.) if descriptor is not None else 0.
            evidence["temporal_appearance"]=round(similarity,3)
            if confirmed and bank and descriptor is not None and similarity<.65:
                pattern = self._registered_pattern(box, ci)
                if pattern is None or pattern['similarity'] < .82 \
                        or self._registered_appearance_similarity(box,ci)<.52:
                    score_cache[box] = (-1., "与已确认物体的外观不一致", evidence)
                    return -1., "与已确认物体的外观不一致"
                evidence['registered_pattern_recovery'] = round(pattern['similarity'],3)
            # A registered background prototype may resemble a transparent
            # object. Bridge only an already confirmed identity with reliable
            # motion and a matching current appearance; color/negative gates
            # are still enforced independently.
            reliable_motion = motion is not None and motion.inlier_ratio>=.75 \
                              and motion.coverage[0]>=.2 and motion.coverage[1]>=.25
            if confirmed and evidence['reason'].startswith('颜色门') \
                    and (reliable_motion or similarity>=.85):
                restored=self._trusted_white_colour_score(box,ci,evidence)
                if restored is not None:
                    score=restored
            strong_visual = similarity>=.9 and int(tgt.get("fail",0))<=3
            if confirmed and evidence["reason"].startswith("背景门") \
                    and ((reliable_motion and similarity>=.8) or strong_visual):
                direct=evidence["without_background_score"]
                if direct>=keep_th:
                    evidence["reason"]="注册背景相近，运动与外观复核通过"
                    evidence["score"]=round(direct,3)
                    score=direct
            score_cache[box] = (score, evidence["reason"], evidence)
            return score, evidence["reason"]

        # Between boundary refreshes, reliable LK motion avoids re-segmenting
        # a blurred silhouette. Appearance gates still run on every frame.
        predicted_score, predicted_reason = None, ""
        async_mode = self._async_search is not None
        if correlation_box is not None and self._near(correlation_box,b,rel=.75):
            correlation_score,_=score_box(correlation_box)
            pattern=self._registered_pattern(correlation_box,ci)
            record=self._diag_localized.get(ci,{})
            supported=self._new_white_identity_ok(correlation_box,ci,correlation_score)
            if not supported and pattern is not None and pattern['similarity']>=.68 \
                    and correlation_score>=self.args.scan_cos_th+.08 \
                    and self._registered_appearance_similarity(correlation_box,ci)>=.50:
                supported=all(sum(float(v)>=0 for v in record.get(name,[]))>=2
                              for name in ('color','background','negative'))
            if correlation_score>=keep_th and supported \
                    and self._localized_geometry_ok(correlation_box,ci) and self._shape_gate_ok(correlation_box,ci,True) \
                    and self._cs_gate_ok(correlation_box,.24):
                tgt['_correlation_frame']=self.frame_idx
                tgt['_correlation_box']=tuple(correlation_box)
                tgt['identity_check_frame']=self.frame_idx-12
                if self._accept_localized_target(tgt,correlation_box,correlation_score,'相关定位复核',False):
                    return
                if not any(t is tgt for t in self.scan_targets):
                    return
                if tgt.get('boundary_pending'):
                    return
        if (motion is not None and ((confirmed and not refresh_due and (age < 8 or strong_motion))
                                  or async_mode)) or template_relocation is not None:
            predicted_score, predicted_reason = score_box(predicted)
            template_ok=template_relocation is None or (
                self._localized_geometry_ok(predicted,ci) and self._shape_gate_ok(predicted,ci,True)
                and self._cs_gate_ok(predicted,.24)
                and (not confirmed or self._diag_localized[ci].get("temporal_appearance",0.)>=.8))
            if predicted_score >= required and template_ok:
                accepted=self._accept_localized_target(tgt, predicted, predicted_score,
                                               "近期画面重新定位" if template_relocation is not None else "光流跟踪",
                                               False, motion)
                # Identity refusal must hide the formal box, but need not freeze
                # an independently reliable position at an increasingly old
                # image. Advance only its private motion reference, without
                # refreshing identity, confirmation or segmentation deadlines.
                if not accepted and any(t is tgt for t in self.scan_targets) \
                        and confirmed and motion is not None \
                        and tuple(predicted)==tuple(motion.box) \
                        and motion.inlier_ratio>=.80 \
                        and motion.coverage[0]>=.20 and motion.coverage[1]>=.25 \
                        and time.perf_counter()-tgt.get('last_verified_time',0.)<=.25 \
                        and predicted_score>=lock_th:
                    pattern=self._registered_pattern(predicted,ci)
                    if pattern is not None and pattern['similarity']>=.68 \
                            and self._registered_appearance_similarity(predicted,ci)>=.50:
                        tgt['box'],tgt['ema']=predicted,None
                        tgt['motion_gray'],tgt['motion_points']=gray,motion.points
                        tgt['motion_corners']=motion.corners
                        tgt['motion_capture_time']=capture_time
                        tgt['motion_ref_age']=0
                        tgt['velocity']=motion.velocity
                        tgt['hidden_motion_frame']=self.frame_idx
                        tracker=tgt.get('_position_tracker')
                        if tracker is not None and _iou(tracker.last_box,predicted)<.5:
                            tgt.pop('_position_tracker',None)
                return

        bw, bh = predicted[2]-predicted[0], predicted[3]-predicted[1]
        margin = .3 + .2*min(3, int(tgt.get("fail", 0)))
        search = (max(0, predicted[0]-int(margin*bw)), max(0, predicted[1]-int(margin*bh)),
                  min(self.W, predicted[2]+int(margin*bw)), min(self.H, predicted[3]+int(margin*bh)))
        tgt["last_boundary_attempt"] = self.frame_idx
        fitted = None if async_mode else fit_bright_component(self.frame, search, self.data.class_ar(ci))
        reason = "未找到完整物体边界"
        if fitted is not None:
            box, occupancy = fitted
            score, gate_reason = score_box(box)
            ratio = ((box[2]-box[0])*(box[3]-box[1])) / max(1., bw*bh)
            if motion is not None and motion.inlier_ratio >= .75:
                consistent = .8 <= ratio <= 1.25 and _iou(box, predicted) >= .8
            else:
                consistent = not confirmed or (.65 <= ratio <= 1.6
                               and _iou(box, predicted) >= (.65 if motion is not None else .5))
            if occupancy >= .45 and .35 <= ratio <= 2.8 and consistent and score >= required \
                    and self._localized_geometry_ok(box, ci) \
                    and self._near(box, predicted, rel=.75) and self._shape_gate_ok(box, ci, True):
                self._accept_localized_target(tgt, box, score, "贴合跟踪", True, motion, occupancy)
                return
            reason = gate_reason if gate_reason != "通过" else f"贴合核验失败({score:.0%})"
        # Only previously confirmed tracks may bridge a few failed segmentations.
        if confirmed and motion is not None and (age < 8 or strong_motion):
            if predicted_score is None:
                predicted_score, predicted_reason = score_box(predicted)
            if predicted_score >= keep_th:
                self._accept_localized_target(tgt, predicted, predicted_score,
                                               "运动边界保持", False, motion)
                return
            if predicted_reason != "通过":
                reason = predicted_reason
        # A sharply matching stored outline can restore sparse motion points
        # after a blur, without assigning that identity to a new scene region.
        if confirmed and motion is None and int(tgt.get("fail",0))<=3:
            visual_score, visual_reason = score_box(predicted)
            match=self._diag_localized[ci].get("temporal_appearance",0.)
            if visual_score>=keep_th and match>=.9:
                self._accept_localized_target(tgt,predicted,visual_score,"外观重新定位",False)
                return
        if async_mode and not confirmed and motion is None:
            score, reason = score_box(predicted)
            previous = tgt.get("candidate_appearance")
            current = describe_appearance(self.frame,predicted)
            if score>=lock_th and previous is not None and current is not None and float(previous@current)>=.9:
                self._accept_localized_target(tgt,predicted,score,"当前帧候选复核",False)
                return
        tgt["fail"] = int(tgt.get("fail", 0)) + 1
        tgt["pending_reason"] = reason
        tgt["boundary_age"] = age + 1
        if not confirmed and tgt["fail"]>2:
            tgt["pend"] = 0
        holding=confirmed and time.perf_counter()-tgt.get("last_verified_time",0.)<=.20
        if holding and motion is not None and tgt["fail"] <= 6:
            tgt["box"], tgt["ema"] = predicted, None
            tgt["motion_gray"], tgt["motion_points"] = gray, motion.points
            tgt["motion_ref_age"] = 0
            tgt['motion_capture_time'] = capture_time
            tgt["motion_corners"] = motion.corners
            tgt["velocity"] = motion.velocity
            tgt["coasting"] = True
        else:
            # Keep the last sharp reference briefly so one blurred frame does
            # not permanently discard all motion points.
            tgt["motion_ref_age"] = int(tgt.get("motion_ref_age", 0)) + 1
            if tgt["motion_ref_age"] > (8 if confirmed else 3):
                tgt["motion_gray"], tgt["motion_points"] = gray, None
        self._diag_reject[ci] = ("移动保持·" if tgt["coasting"] else "") + reason
        expired=confirmed and time.perf_counter()-tgt.get("last_verified_time",0.) > (.6 if trusted_identity else .35)
        if expired or tgt["fail"] > ((18 if trusted_identity else 10) if confirmed else 5):
            self.scan_targets.remove(tgt)

    def _run_background_search(self, job):
        # This App and its learning snapshot are owned only by the worker.
        started=time.perf_counter()
        if self._search_app is None or self._search_app._job_epoch!=job["epoch"]:
            self._search_app=App(copy.deepcopy(self.args),self.backbone,job["data"],
                                 PefslClassifier(self.clf.ctype,self.clf.k),self.R)
            self._search_app._job_epoch=job["epoch"]
            self._search_app.mode="B"
        app=self._search_app
        app._in_search_worker=True
        app._generic_identity_fast=True
        app._generic_full_checks_left=2
        app._generic_job_deadline=started+.16
        app._generic_job_queries={}
        app._generic_template_preferences=dict(job.get('template_preferences',{}))
        app.frame[:]=job["frame"]
        app.raw_frame=job["frame"]
        app._capture_time=job['source_time']
        app._diag_localized={}
        app._diag_reject={}
        app._diag_colour_candidates={}
        app._diag_generic_candidates={}
        app._diag_generic_source_work={}
        app.bg_model=job["background"]
        app.frame_idx=(job["number"]+1)*12*max(1,app.args.scan_interval)
        app.scan_targets=copy.deepcopy(job["targets"])
        proposals=[]
        if job["kind"]=="local":
            local_targets=app.scan_targets
            if len(local_targets)>1 and any(app._generic_class(t['cls']) for t in local_targets):
                start=getattr(app,'_local_search_cursor',0)%len(local_targets)
                local_targets=local_targets[start:]+local_targets[:start]
                app._local_search_cursor=(start+1)%len(local_targets)
            generic_deadline=app._generic_job_deadline
            for target in local_targets:
                b=target["box"];ci=target["cls"]
                w,h=b[2]-b[0],b[3]-b[1]
                if target.get("ever_confirmed"):
                    steps=max(1.,min(6.,job.get('prediction_steps',1.)))
                    velocity=tuple(v*steps for v in target.get("velocity",(0.,0.)))
                    search=recovery_region(b,velocity,app.W,app.H,target.get("fail",0))
                else:
                    search=(max(0,b[0]-int(.3*w)),max(0,b[1]-int(.3*h)),
                            min(app.W,b[2]+int(.3*w)),min(app.H,b[3]+int(.3*h)))
                if app._generic_class(ci):
                    # Generic recovery uses the same class-independent sources
                    # as acquisition, restricted to the predicted vicinity.
                    # It must never fall through to the legacy bright rules.
                    remaining=generic_deadline-time.perf_counter()
                    if remaining>=.025:
                        proposals.extend(app._search_generic_proposals(classes=[ci],region=search,budget=remaining))
                    if proposals:break
                    continue
                fitted=bright_component_candidates(app.frame,search,app.data.class_ar(ci),
                                                  max_candidates=6)
                complete=False
                threshold=app.args.scan_cos_th-.14 if target.get("ever_confirmed") else app.args.scan_cos_th+.02
                recovering=bool(target.get('ever_confirmed') and target.get('fail',0)>0)
                verifying_new=bool(not target.get('ever_confirmed') and target.get('pend',0)>0
                                   and target.get('boundary_votes',0)>0)
                # During recovery try measured components before expensive
                # boundary variants. Only a complete learned object can take
                # this shortcut; otherwise keep the original full refinement.
                # A newly acquired identity also needs its second independent
                # segmentation promptly. Try complete measured components first;
                # current identity/endpoints and all admission gates still apply.
                for quick in ((True,False) if recovering or verifying_new else (False,)):
                    candidates=fitted if quick else app._refine_bright_boundaries(fitted,ci)
                    boxes=[box for box,occupancy in candidates]
                    scores=app._score_localized_boxes(boxes)[:,ci] if boxes else []
                    for j in app._localized_candidate_order(boxes,scores,ci,threshold):
                        box,occupancy=candidates[j]
                        score=float(scores[j])
                        if not app._localized_boundary_consistent(box,target,score):
                            continue
                        if occupancy<.45 or score<threshold \
                                or not app._localized_geometry_ok(box,ci) \
                                or not app._shape_gate_ok(box,ci,True):
                            continue
                        if quick and (not endpoint_boundary_support(app.frame,box)[1]
                                      or not app._new_white_identity_ok(box,ci,score)):
                            continue
                        proposals.append(dict(cls=ci,box=box,occupancy=occupancy))
                        complete=True
                        break
                    if complete:
                        break
                if not complete and app._new_white_identity_ok(b,ci,threshold) \
                        and app._localized_geometry_ok(b,ci) and app._shape_gate_ok(b,ci,True):
                    # Transparent tails can fail segmentation; use the learned
                    # object appearance as independent evidence for this box.
                    proposals.append(dict(cls=ci,box=b,occupancy=target.get("boundary_occupancy"),
                                          registered_appearance=True))
        else:
            generic_classes={ci for ci in range(len(app.data.labels)) if app._generic_class(ci)}
            legacy_classes=set(range(len(app.data.labels)))-generic_classes
            mixed=bool(generic_classes and legacy_classes)
            # A legacy multi-scale scan must not hold a completed generic
            # proposal until it expires. Alternate independent current-frame
            # jobs when both policies are present in one learning archive.
            # Local recovery jobs must not consume a global policy's turn.
            # Otherwise every global job can share one parity and starve the
            # other policy indefinitely while local jobs fill the gaps.
            global_turn=getattr(app,'_mixed_global_turn',0)
            run_generic=not mixed or global_turn%2==0
            if mixed:app._mixed_global_turn=global_turn+1
            run_legacy=not mixed or not run_generic
            remaining=app._generic_job_deadline-time.perf_counter()
            proposals=app._search_generic_proposals(budget=max(0.,remaining)) if run_generic else []
            generic_covered={p['cls'] for p in proposals}
            if run_legacy:
                proposals.extend(app._search_colour_proposals(exclude_classes=generic_classes|generic_covered))
            covered={p['cls'] for p in proposals}
            # Rejected colour components must not suppress independent search.
            # Every fallback is still verified on the current camera frame.
            if run_legacy and legacy_classes-covered:
                if mixed:app._scan_tick(search_only=True,search_classes=legacy_classes)
                else:app._scan_tick(search_only=True)
                for target in app.scan_targets:
                    if target['cls'] not in covered and target.get("verified") \
                            and target.get("last_evidence_frame")==app.frame_idx:
                        proposals.append(dict(cls=target["cls"],box=target["box"],
                                              occupancy=target.get("boundary_occupancy")))
            for ci,records in app._diag_colour_candidates.items():
                if ci not in covered and records:
                    app._diag_reject[ci]='完整彩色候选·'+records[0]['reason']+'·'+records[0]['evidence']['reason']
        gray=cv2.cvtColor(app.frame,cv2.COLOR_BGR2GRAY)
        for p in proposals:
            if app._generic_class(p['cls']):
                hint=app._registered_pattern(p['box'],p['cls'])
                geometry={} if hint is None else hint.get('geometry',{})
                method=geometry.get('detector');index=geometry.get('reference')
                if method in ('orb','sift') and isinstance(index,(int,np.integer)):
                    references=app.data.class_scene_refs(p['cls'],method)
                    if 0<=index<len(references):
                        hint=dict(hint,_hint_scene_ref=references[index],_hint_scene_method=method)
                        entry=references[index]
                view_index=geometry.get('reference_view')
                orientation={} if hint is None else hint.get('spatial',hint.get('dense',{}))
                if view_index is None:view_index=orientation.get('reference')
                if view_index is None and hint is not None and hint.get('_hint_pattern') is not None:
                    for group,patterns in app.data._generic_pattern_groups.get(p['cls'],{}).items():
                        if any(v is hint['_hint_pattern'] for v in patterns):
                            view_index=group[0];orientation=dict(flip=group[1],rotation=group[2]);break
                if view_index is None and method in ('orb','sift') and isinstance(index,(int,np.integer)):
                    references=app.data.class_scene_refs(p['cls'],method)
                    if 0<=index<len(references) and len(references[index])>3:view_index=int(references[index][3])
                if view_index is not None:
                    ci=p['cls'];hint=dict(hint)
                    colour_refs=app.data.class_colour_geometry_refs(ci)
                    selected=next((v for v in colour_refs if len(v)>3 and v[3]==view_index),None)
                    if selected is not None:hint['_hint_colour_ref']=selected
                    cache=getattr(app,'_selected_reference_checks',{})
                    flipped=bool(orientation.get('flip',False));rotation=int(orientation.get('rotation',0))
                    hint['_hint_reference_pose']=(flipped,rotation)
                    key=(ci,view_index,flipped,rotation)
                    if key not in cache:
                        view=app.data.class_reference_views(ci)[view_index]
                        view=np.ascontiguousarray(np.rot90(view[:,::-1] if flipped else view,rotation))
                        dense=dense_reference_features([view]);spatial=spatial_reference_features([view])
                        reference_gray=cv2.GaussianBlur(cv2.resize(cv2.cvtColor(view,cv2.COLOR_BGR2GRAY),(96,96)),(3,3),.6)
                        points=cv2.goodFeaturesToTrack(reference_gray,maxCorners=60,qualityLevel=.02,minDistance=6,blockSize=3)
                        cache[key]=(dense[0] if dense else None,spatial[0] if spatial else None,
                            dict(gray=reference_gray,points=points))
                    app._selected_reference_checks=cache
                    dense,spatial,point_ref=cache[key]
                    hint['_hint_point_ref']=point_ref
                    if spatial is not None and (hint.get('_hint_spatial_ref') or {}).get('patch_size')!=16:
                        hint['_hint_spatial_ref']=spatial
                    if dense is not None and hint.get('full_quad') is not None:
                        hint['_hint_dense_ref']=dense
                        hint['_hint_dense_seed']=dict(dense={k:dense[k] for k in ('reference','rotation','flip')},
                            full_quad=hint['full_quad'])
                p['identity_hint']=hint
            p["points"]=(app._seed_generic_motion(p['box'],p['cls'],gray) if app._generic_class(p['cls'])
                         else seed_points(app.frame,gray,p['box']))
            p["appearance"]=describe_appearance(app.frame,p["box"])
        return dict(epoch=job["epoch"],kind=job["kind"],source_time=job["source_time"],
                    # Completed LEARNING references only. Publishing a bank
                    # never constructs it or exports camera correspondences.
                    dense_learning_refs={ci:refs for ci,refs in app.data._dense_refs.items()
                        if app._generic_class(ci)},
                    fine_learning_refs={key[0]:refs for key,refs in app.data._spatial_refs.items()
                        if isinstance(key,tuple) and key[1]==16 and app._generic_class(key[0])},
                    visual_learning_refs={ci:refs for ci,refs in app.data._visual_refs.items()
                        if app._generic_class(ci)},
                    source_frame=job.get('source_frame'),
                    gray=gray,proposals=proposals,stage=getattr(app,"_diag_stage",None),
                    rejection=getattr(app,"_diag_reject",{}).copy(),
                    generic_candidates=copy.deepcopy(getattr(app,'_diag_generic_candidates',{})),
                    generic_source_work=copy.deepcopy(getattr(app,'_diag_generic_source_work',{})),
                    colour_candidates=copy.deepcopy(getattr(app,'_diag_colour_candidates',{})))

    def _apply_search_result(self, result, whites):
        # Do not let a later failed crop displace the current track's
        # successful task selection. Keep candidate checks and all their
        # current-frame evidence unchanged; only restore the next selection.
        next_tasks={}
        for target in self.scan_targets:
            ci=target['cls'];state=self._identity_task_states.get(ci,{})
            if (target.get('ever_confirmed') and target.get('verified')
                    and self._generic_class(ci) and state.get('status')=='verified'
                    and state.get('frame')==self.frame_idx
                    and tuple(state.get('box',()))==tuple(target['box'])
                    and 'next' in state):
                next_tasks[ci]=state['next']
        try:
            return self._apply_search_result_impl(result,whites)
        finally:
            for ci,next_task in next_tasks.items():
                state=self._identity_task_states.get(ci,{})
                if state.get('frame')==self.frame_idx and state.get('status') in ('pending','rejected'):
                    state['next']=next_task

    def _search_failed(self,error):
        # SingleFlightWorker already contains handler exceptions and remains
        # usable. Keep asynchronous identity budgets active after one bad job.
        # Reset only worker-owned search state; learned data and tracks stay.
        self._search_error_streak=min(4,getattr(self,'_search_error_streak',0)+1)
        self._search_last_error=str(error)
        self._search_app=None
        self._search_data_snapshot=None
        self._deferred_search_result=None
        self._next_search_time=time.perf_counter()+min(1.,.125*2**(self._search_error_streak-1))
        print('[search] job failed; bounded retry:',self._search_last_error,flush=True)

    def _apply_search_result_impl(self, result, whites):
        now=time.perf_counter()
        epoch=result.get('epoch')
        if not isinstance(epoch,(int,np.integer)) or epoch!=self._search_epoch:
            return
        if result.get("error"):
            self._search_failed(result['error'])
            return
        try:
            validate_search_result(result,self.W,self.H,len(self.data.labels),APPEARANCE_SIZE)
        except (ValueError,TypeError,OverflowError) as error:
            self._search_failed(error)
            return
        # Completed LEARNING references are independent of camera age and
        # confirmation. The checked epoch binds them to this learning version.
        # Warm at most one missing generic bank per delivery, prioritizing
        # incoming/current targets. Copies own their arrays; no image proof,
        # target state or confirmation vote is imported with them.
        references=result.get('dense_learning_refs',{})
        if isinstance(references,dict) and references:
            priority={p['cls'] for p in result['proposals']} | {t['cls'] for t in self.scan_targets}
            eligible=[ci for ci,refs in references.items()
                if isinstance(ci,(int,np.integer)) and not isinstance(ci,(bool,np.bool_))
                and 0<=ci<len(self.data.labels) and self._generic_class(ci)
                and ci not in self.data._dense_refs and isinstance(refs,list)]
            if eligible:
                ci=min(eligible,key=lambda ci:(ci not in priority,int(ci)))
                self.data._dense_refs[ci]=copy.deepcopy(references[ci])
        # The worker already prepared these immutable LEARNING banks. Copy
        # at most one missing class per delivery, before a current-image
        # matching deadline. Never transport camera proof or confirmation.
        for field,cache,fine in (
                ('fine_learning_refs',self.data._spatial_refs,True),
                ('visual_learning_refs',self.data._visual_refs,False)):
            references=result.get(field,{})
            if not isinstance(references,dict):continue
            eligible=[ci for ci,refs in references.items()
                if isinstance(ci,(int,np.integer)) and not isinstance(ci,(bool,np.bool_))
                and 0<=ci<len(self.data.labels) and self._generic_class(ci)
                and ((ci,16) if fine else ci) not in cache and isinstance(refs,list)]
            if eligible:
                ci=min(eligible,key=lambda value:(value not in {p['cls'] for p in result['proposals']},int(value)))
                cache[(ci,16) if fine else ci]=copy.deepcopy(references[ci])
        self._search_last_ms=result['elapsed_ms']
        self._search_error_streak=0
        self._search_last_error=None
        if not result.get('_continuation'):
            self._search_times.append(now)
        while self._search_times and now-self._search_times[0]>2.:
            self._search_times.popleft()
        self.search_fps=((len(self._search_times)-1)/(now-self._search_times[0])) if len(self._search_times)>1 else 0.
        self._search_age_ms=(now-result["source_time"])*1000
        self._search_last_kind=result['kind']
        if result.get('colour_candidates'):
            self._diag_colour_candidates=result['colour_candidates']
            self._diag_colour_source_time=result['source_time']
        self._diag_generic_candidates=result.get('generic_candidates',{})
        self._diag_generic_source_work=result.get('generic_source_work',{})
        self._diag_generic_source_time=result['source_time']
        self._diag_generic_source_frame=result.get('source_frame')
        if self._search_age_ms>600:
            self._search_dropped_age=getattr(self,'_search_dropped_age',0)+1
            for proposal in result['proposals']:
                ci=proposal['cls']
                if not any(t['cls']==ci and t.get('verified') for t in self.scan_targets):
                    self._diag_reject[ci]=(f'后台交接：候选已超过600ms有效期 '
                        f'age_ms={self._search_age_ms:.1f} source_frame={result.get("source_frame")} '
                        f'current_frame={self.frame_idx}')
            return
        if result["kind"]=="global":
            if result["stage"] is not None:
                self._diag_stage=result["stage"]
            for ci,reason in result["rejection"].items():
                if not any(t["cls"]==ci and t.get("verified") for t in self.scan_targets):
                    self._diag_reject[ci]=reason
        gray=None
        generic_processed=0
        for proposal_index,p in enumerate(result["proposals"]):
            ci=p["cls"]
            old=next((t for t in self.scan_targets if t["cls"]==ci),None)
            new_target=old is None
            original=old
            if old is not None and self._generic_class(ci):old=dict(old)
            with self._generic_reference_trial(ci,protect_choice=bool(original and original.get('ever_confirmed'))) as commit_reference:
                stale = self._search_age_ms > 350
                stale_new = stale and (old is None or not old.get('ever_confirmed'))
                if result["kind"]=="global" and old is not None and old.get("verified") and old["pend"]>=4:
                    continue
                if self._generic_class(ci):
                    if generic_processed>=1:
                        # Keep remaining source-frame proposals for the next
                        # frame. Their epoch and original expiry still apply.
                        self._deferred_search_result=dict(result,
                            proposals=result['proposals'][proposal_index:],_continuation=True)
                        break
                    generic_processed+=1
                    if p.get('identity_hint') is not None:
                        self._remember_generic_hint(ci,p['identity_hint'])
                        # Tracking may have cached failure before this newly
                        # selected learned template arrived in the same frame.
                        if getattr(self,'_pattern_cache_key',None)==(self.frame_idx,self._search_epoch):
                            for cached_key in list(self._pattern_cache):
                                if cached_key[0]==ci:self._pattern_cache.pop(cached_key,None)
                if gray is None:
                    gray=cv2.cvtColor(self.frame,cv2.COLOR_BGR2GRAY)
                motion=track_points(result["gray"],gray,p["points"],p["box"])
                if self._generic_class(ci):motion=self._supported_generic_motion(motion,p['box'])
                # Sparse flow selects a location but cannot certify an old
                # ROI. Apply the same CURRENT acquisition as when flow is
                # absent, retaining source expiry and all identity gates.
                stale_reacquisition=stale and self._generic_class(ci) and (
                    motion is None or motion.inlier_ratio<.9
                    or motion.coverage[0]<.3 or motion.coverage[1]<.3)
                if stale and (motion is None or motion.inlier_ratio < .8
                              or motion.coverage[0]<.2 or motion.coverage[1]<.25) and not stale_reacquisition:
                    self._diag_reject[ci]='后台交接：旧候选运动证据不足'
                    continue
                box=motion.box if motion is not None else p["box"]
                fitted=self._fit_generic_reference(box,ci,
                    allow_projection=not self._foreground_generic_identity())
                if fitted is not None:box=fitted[0]
                if self._generic_class(ci) and p.get('identity_hint') is not None:
                    cached=getattr(self,'_localized_score_cache',{}).get(tuple(box))
                    if cached is not None and cached[0][ci]<0 \
                            and cached[1][ci].get('reason','').startswith('颜色门'):
                        self._localized_score_cache.pop(tuple(box),None)
                if stale:
                    pattern=self._registered_pattern(box,ci)
                    current_geometry=False
                    current_identity=False
                    if self._generic_class(ci):
                        current_score=float(self._score_localized_boxes([box])[0,ci])
                        current_geometry=self._generic_geometry_admission(box,ci,current_score)
                        # Re-establish identity from CURRENT pixels. A fresh-acquisition
                        # candidate must pass this full acquisition too; no old
                        # score extends the 600 ms result lifetime.
                        current_identity=bool((stale_reacquisition or (motion is not None
                            and motion.inlier_ratio>=.9 and motion.coverage[0]>=.3 and motion.coverage[1]>=.3))
                            and current_score>=self.args.scan_cos_th+.02
                            and self._new_white_identity_ok(box,ci,current_score))
                    if stale_reacquisition and not current_identity:
                        state=self._identity_task_states.get(ci,{})
                        pending=(current_score>=self.args.scan_cos_th+.02
                            and state.get('status')=='pending' and state.get('frame')==self.frame_idx
                            and tuple(state.get('box',()))==tuple(box))
                        age=(time.perf_counter()-result['source_time'])*1000.
                        reason=('当前分类门未通过' if current_score<self.args.scan_cos_th+.02 else
                            '当前身份核验待继续' if pending else '当前身份核验未通过')
                        if age>600:reason='核验后候选已超过600ms有效期'
                        self._diag_reject[ci]=(f'后台交接：{reason} age_ms={age:.1f} '
                            f'source_frame={result.get("source_frame")} current_frame={self.frame_idx} '
                            f'box={tuple(box)}')
                        if pending and age<=600:
                            # Continue task selection, never proof or source age.
                            # Give other untried classes the next frame's turn.
                            remaining=result['proposals'][proposal_index:]
                            self._deferred_search_result=dict(result,
                                proposals=remaining[1:]+remaining[:1],_continuation=True)
                            return
                        continue
                    if (pattern is None or pattern['similarity']<(.82 if stale_new else .78)) and not (current_geometry or current_identity):
                        continue
                    # A fresh-acquisition proposal needs a complete current acquisition;
                    # the existing motion-supported route keeps its own gates.
                    if stale_new and (motion is None or motion.inlier_ratio < .85) \
                            and not (current_geometry or (stale_reacquisition and current_identity)):
                        continue
                # Fitting may validate the current crop and return None when
                # its boundary needs no correction. For a NEW acquisition,
                # keep that current-frame proof ahead of source-image HOG and
                # template heuristics; all admission gates below still apply.
                measured_identity=stale_reacquisition and current_identity
                if old is None and self._generic_class(ci) and motion is None:
                    cache=getattr(self,'_pattern_cache',{})
                    current=(cache.get((ci,tuple(box))) if getattr(self,'_pattern_cache_key',None)
                        ==(self.frame_idx,self._search_epoch) else None)
                    if current is not None:
                        measured_identity=self._registered_identity_ok(
                            self._registered_appearance_similarity(box,ci),current,ci)
                template_match=None
                if motion is None:
                    direct=describe_appearance(self.frame,box)
                    same=direct is not None and p["appearance"] is not None and float(p["appearance"]@direct)>=.9
                    if not same and not measured_identity:
                        template_match=relocate_template(result["gray"],gray,p["box"])
                        if template_match is not None:
                            box=template_match[0]
                if motion is None:
                    appearance=describe_appearance(self.frame,box)
                    if template_match is None and not measured_identity and (p["appearance"] is None or appearance is None
                            or float(p["appearance"]@appearance)<.9):
                        current_score=float(self._score_localized_boxes([box])[0,ci]) if self._generic_class(ci) else -1.
                        if not self._generic_geometry_admission(box,ci,current_score):
                            continue
                if p.get("registered_appearance") and not self._new_white_identity_ok(box,ci,self.args.scan_cos_th+.02):
                    continue
                if whites[ci] or p.get('colour_boundary') or p.get('generic_boundary'):
                    score=float(self._score_localized_boxes([box])[0,ci])
                    self._diag_localized[ci]=self._last_localized_evidence[0][ci]
                    evidence=self._diag_localized[ci]
                    # The same registered-background ambiguity bridge as tracking:
                    # only a known identity, reliable motion and matching appearance.
                    if old is not None and old.get("ever_confirmed") and result["kind"]=="local" \
                            and evidence["reason"].startswith("背景门") and motion is not None \
                            and motion.inlier_ratio>=.75 and motion.coverage[0]>=.2 and motion.coverage[1]>=.25:
                        descriptor=describe_appearance(self.frame,box)
                        similarity=max((float(descriptor@v) for v in old.get("appearance_bank",[])),default=0.) \
                                   if descriptor is not None else 0.
                        if similarity>=.8 and evidence["without_background_score"]>=self.args.scan_cos_th-.14:
                            score=evidence["without_background_score"]
                            evidence.update(score=round(score,3),temporal_appearance=round(similarity,3),
                                            reason="注册背景相近，当前外观与运动复核通过")
                    if whites[ci] and not self._localized_geometry_ok(box,ci):
                        continue
                    new_identity=old is None or not self._near(box,old["box"],rel=.75)
                    if new_identity and self.data.class_visual_refs(ci):
                        reference=self._registered_appearance_similarity(box,ci)
                        evidence["registered_appearance"]=round(reference,3)
                        # One saved thumbnail represents only one learned view.
                        # Use it to reject marginal new identities; do not average
                        # unrelated score scales or veto stronger learned views.
                        if not self._new_white_identity_ok(box,ci,score):
                            pending=self._identity_pending(ci)
                            retry=(self._generic_class(ci) and not pending
                                and not p.get('current_identity_retried')
                                and score>=self.args.scan_cos_th+.02)
                            if (pending or retry) and self._search_age_ms<350:
                                evidence['reason']='身份核验待继续' if pending else '当前身份失败，待下一帧复核一次'
                                self._diag_reject[ci]=evidence['reason']
                                remaining=result['proposals'][proposal_index:]
                                if retry:
                                    # A completed rejection applies to this
                                    # image, not every future image. Keep only
                                    # one position/reference retry, behind
                                    # untried candidates. All original epoch,
                                    # source age and current-image gates apply.
                                    remaining=remaining[1:]+[dict(p,current_identity_retried=True)]
                                self._deferred_search_result=dict(result,
                                    proposals=remaining,_continuation=True)
                                return
                            pattern=self._registered_pattern(box,ci)
                            evidence['current_identity']=dict(
                                feature_score=round(score,3),reference=round(reference,3),
                                pattern_present=pattern is not None,
                                pattern_similarity=None if pattern is None else pattern.get('similarity'),
                                dense={} if pattern is None else pattern.get('dense',{}),
                                geometry={} if pattern is None else pattern.get('geometry',{}),
                                spatial={} if pattern is None else pattern.get('spatial',{}))
                            evidence["reason"]="当前帧学习身份复核未通过"
                            self._diag_reject[ci]=evidence["reason"]
                            continue
                else:
                    crop=crop_resize(self.frame,box,self.in_size)[None]
                    features=self.backbone(preprocess_batch(crop))
                    scores=self.clf.cosine_batch(features,self.data)
                    scores=self._hist_gate(crop,scores)
                    scores=self._bg_gate(features,scores)
                    score=float(self._neg_gate(features,scores)[0,ci])
                required=self.args.scan_cos_th+.02
                if result["kind"]=="local" and old is not None and old.get("ever_confirmed") \
                        and motion is not None and motion.inlier_ratio>=.75:
                    required=self.args.scan_cos_th-.14
                if score<required and not self._generic_geometry_admission(box,ci,score):
                    self._diag_reject[ci]=(f'当前帧分类门未通过 score={score:.3f} required={required:.3f}')
                    continue
                if not self._shape_gate_ok(box,ci,whites[ci]):
                    self._diag_reject[ci]='当前帧形状核验未通过'
                    continue
                if not self._cs_gate_ok(box,cs_th=.24 if whites[ci] else .16):
                    self._diag_reject[ci]='当前帧中心与周边对比核验未通过'
                    continue
                if old is not None:
                    ob=old["box"]
                    ratio=((box[2]-box[0])*(box[3]-box[1]))/max(1.,(ob[2]-ob[0])*(ob[3]-ob[1]))
                    reliable=old.get("motion_quality",0.)>=.75
                    size_ok=.65<=ratio<=1.25 if reliable else .65<=ratio<=1.6
                    if old.get("ever_confirmed") and old.get("fail",0)<=4 \
                            and not self._localized_boundary_consistent(box,old,score):
                        continue
                    if result["kind"]=="local" and not self._localized_boundary_consistent(box,old,score):
                        continue
                    if not old.get("ever_confirmed") and old.get("boundary_votes",0)>0 \
                            and result["kind"]=="global" and (not size_ok or _iou(box,ob)<.65):
                        continue
                    if not self._near(box,ob,rel=.75):
                        old.update(pend=0,ever_confirmed=False,appearance_bank=[],velocity=(0.,0.),ema=None,
                                   boundary_votes=0,candidate_started=now)
                else:
                    if any(_same_obj(box,t["box"]) for t in self.scan_targets):
                        continue
                    old=dict(cls=ci,label=self.data.labels[ci],box=box,pend=0,ema=None,ever_confirmed=False,
                             candidate_started=now)
                old["candidate_appearance"]=describe_appearance(self.frame,box)
                boundary=result["source_time"]>old.get("last_boundary_source_time",-1.)
                if boundary:
                    old["last_boundary_source_time"]=result["source_time"]
                if p.get("registered_appearance"):
                    # A hidden new identity has not yet supplied a complete
                    # segmentation. Template-only results cannot supply its first
                    # independent boundary vote.
                    if 'segmentation_frame' in old:
                        old["boundary_votes"]=old.get("boundary_votes",0)+1
                    boundary=False
                if stale_reacquisition and time.perf_counter()-result['source_time']>.6:
                    self._diag_reject[ci]='后台交接：当前核验完成时已超过600ms有效期'
                    continue
                accepted=self._accept_localized_target(old,box,score,"异步搜索·当前帧核验",
                    boundary,occupancy=p["occupancy"])
                if (new_target and accepted and self._generic_class(ci)
                        and p.get('identity_hint') is not None and self._search_age_ms<=350):
                    old['_generic_acquisition_source']=dict(gray=result['gray'],box=tuple(p['box']),
                        time=result['source_time'],epoch=result['epoch'])
                if accepted and p.get('proposal_source'):
                    old['proposal_source']=p['proposal_source']
                if accepted and p.get('template_variant') is not None:
                    old['template_variant']=tuple(p['template_variant'])
                if accepted:
                    commit_reference()
                    if original is not None and self._generic_class(ci):
                        original.clear();original.update(old)
                        getattr(self,'_generic_motion_cache',{}).pop(id(original),None)
                # Admission can reject before initializing scores/motion fields.
                # A strong current identity may enter as a fully initialized HIDDEN
                # boundary candidate. It has no votes or formal display privileges.
                if new_target and (accepted or old.get('admission_pending')):
                    self.scan_targets.append(old)

    def _async_search_tick(self, healthy_classes, whites):
        if not hasattr(self,"_diag_reject"):
            self._diag_reject={}
        now=time.perf_counter()
        while self._search_times and now-self._search_times[0]>2.:
            self._search_times.popleft()
        self.search_fps=(len(self._search_times)-1)/(now-self._search_times[0]) if len(self._search_times)>1 else 0.
        result=self._deferred_search_result
        self._deferred_search_result=None
        if result is None:
            result=self._async_search.poll()
        if result is not None:
            apply_started=time.perf_counter()
            self._apply_search_result(result,whites)
            perf=getattr(self,'_generic_perf',{})
            perf['result_apply_ms']=(time.perf_counter()-apply_started)*1000.
            self._generic_perf=perf
        if self._async_search is None:
            return
        healthy_classes={t["cls"] for t in self.scan_targets if t.get("verified") and t["pend"]>=4 and t.get("fail",0)==0}
        for target in self.scan_targets:
            if (target['cls'] in healthy_classes and target.get('ever_confirmed')
                    and target.get('template_variant') is not None):
                self._generic_template_preferences[target['cls']]=tuple(target['template_variant'])
        localizable=lambda ci: whites[ci] or self._generic_class(ci)
        pending=[t for t in self.scan_targets if localizable(t['cls']) and not t.get("ever_confirmed")]
        def can_recover(t):
            if self._generic_class(t['cls']):
                return now<=t.get('generic_recovery_until',t.get('last_verified_time',0.)+.4)
            return (now-t.get('last_verified_time',0.) <= (
                .5 if now-t.get('strong_identity_time',0.)<=.6 else .25)
                or (t.get('boundary_pending') and 0.<=now-t.get('boundary_pending_since',0.)<=1.))
        recovering=[t for t in self.scan_targets if localizable(t['cls']) and t.get('ever_confirmed')
                    and (t.get('fail',0)>0 or (self._generic_class(t['cls']) and not t.get('verified')))
                    and can_recover(t)]
        local=recovering+pending+[t for t in self.scan_targets if t["cls"] in healthy_classes and localizable(t['cls'])
                       and t.get("boundary_age",0)>=8 and (not self._generic_class(t['cls']) or
                           now>=max(t.get('generic_boundary_time',0.)+.35,
                                    t.get('generic_boundary_retry_after',0.)))]
        missing=set(range(len(self.data.labels)))-healthy_classes
        if not missing and not local:
            return
        now=time.perf_counter()
        if now<self._next_search_time:
            return
        if self._search_data_snapshot is None:
            self._search_data_snapshot=copy.deepcopy(self.data)
        keys=("cls","label","box","asize","ema","cos","gcos","pend","fail","verified",
              "ever_confirmed","boundary_age","motion_quality","last_evidence_frame","boundary_occupancy",
              "velocity","strong_identity_time","appearance_bank","boundary_votes")
        # Pending targets need an independent fresh segmentation. Alternate
        # scene and local verification so a weak local crop cannot deadlock
        # an otherwise valid scene proposal at three confirmation frames.
        global_needed=bool(missing-{t["cls"] for t in pending}) \
                      or bool(pending and self._search_job_count%2==0)
        # Give recent moving targets two local attempts per global search.
        # Periodic global searches still allow recovery beyond the local area.
        if recovering and self._search_job_count%3!=2:
            global_needed=False
        if not recovering and self._search_job_count%2==1 and any(
                self._generic_class(t['cls']) and t['cls'] in healthy_classes
                and t.get('boundary_age',0)>=8 for t in local):
            # Missing classes must not permanently starve a healthy generic
            # object's overdue boundary refresh with global-only searches.
            global_needed=False
        selected=local if not global_needed else [t for t in self.scan_targets
                 if t["cls"] in healthy_classes or localizable(t['cls'])]
        local_subset=False
        if not global_needed and len(selected)>1 \
                and any(self._generic_class(ci) for ci in range(len(self.data.labels))) \
                and any(not self._generic_class(ci) for ci in range(len(self.data.labels))):
            subset_start=getattr(self,'_local_subset_cursor',0)%len(selected)
            selected=[selected[subset_start]]
            local_subset=True
        context=[{k:copy.deepcopy(t[k]) for k in keys if k in t} for t in selected]
        for source,target in zip(selected,context):
            if (self._generic_class(source['cls']) and not source.get('verified')
                    and source.get('generic_hidden_motion')
                    and now-source.get('generic_hidden_motion_time',0.)<=.2
                    and now<=source.get('generic_recovery_until',0.)):
                target['box']=tuple(source['motion_box'])
            elif (self._generic_class(source['cls']) and not source.get('verified')
                    and now-source.get('generic_recovery_box_time',0.)<=.15
                    and now<=source.get('generic_recovery_until',0.)):
                target['box']=tuple(source.get('generic_recovery_box',source['box']))
        job=dict(epoch=self._search_epoch,number=self._search_job_count,
                 kind="global" if global_needed else "local",data=self._search_data_snapshot,
                 source_frame=self.frame_idx,
                 frame=self.frame.copy(),source_time=getattr(self,"_capture_time",now),
                 prediction_steps=max(1.,min(6.,self._search_last_ms*.001*max(5.,float(
                     getattr(self,'_perf_current',{}).get('camera_fps',30.))))),
                 background=None if self.bg_model is None else self.bg_model.copy(),targets=context,
                 # Do not spend a known class's shortcut on a healthy target
                 # or ahead of a missing class without a confirmed preference.
                 template_preferences=({ci:self._generic_template_preferences[ci] for ci in missing}
                     if getattr(self.args,'confirmed_template_reacquire',False)
                     and missing.issubset(self._generic_template_preferences) else {}))
        if self._async_search.submit(job):
            if local_subset:self._local_subset_cursor=subset_start+1
            if not global_needed:
                for target in selected:
                    if self._generic_class(target['cls']) and target['cls'] in healthy_classes:
                        target['generic_boundary_retry_after']=now+.35
            self._search_job_count+=1
            hz=getattr(self.args,"search_hz",8.)
            self._next_search_time=now+1./max(1.,max(hz,12.) if recovering else hz)

    def _scan_tick(self, search_only=False,search_classes=None):
        """多目标自动定位（两级扫描 + 逐目标跟踪，全部绝对余弦门控）：
        每个已学习类别各持有一个锁定目标——多个学习过的物体同时出现时
        各自画框、各自跟踪、互不干扰。
        阶段1 跟踪（每帧/每目标）：只在各自框附近局部细化，中心窗口优先
                    （防漂移）；与全局搜索同一套三重核验(颜色/背景负样本/
                    类别歧义)，不过关走宽限(已确认 3 帧/未确认 1 帧)后释放
                    ——物体被换走或举到空背景时旧锁定不再硬撑；核验赢家
                    换类时标签即时更正。
        阶段2 全局（按节奏；有已确认目标时每 8 帧维护一次）：多尺度粗搜 ->
                    逐类取最佳窗口 -> 精细化网格对齐；新目标连续确认 2 次
                    才正式显示(单帧噪声不画框)；与其它类目标高度重叠的窗口
                    丢弃(同一物体不重复框)。
        窗口边长封顶 72% 短边：整屏大窗是'背景冒充物体'的载体。
        迟滞双门槛：新锁定 >= th+0.02，保持 >= th-0.14。新锁定还需通过
        '中心-周围对比'门控(桌面/墙面等均匀背景被直接拒绝)。"""
        if not search_only:
            self._diag_localized = {}
            previous_frame=getattr(self,'_diag_reject_frame',self.frame_idx-1)
            history=getattr(self,'_diag_reject_history',{})
            for ci,reason in getattr(self,'_diag_reject',{}).items():
                history[ci]=dict(frame=previous_frame,reason=reason)
            self._diag_reject_history=history
            self._diag_reject={}
            self._diag_reject_frame=self.frame_idx
        th = self.args.scan_cos_th
        th_lock = th + 0.02      # 新锁定门槛（默认 0.80：过滤低置信度背景误检）
        th_keep = th - 0.14       # 跟踪保持门槛(比锁定门槛低一档宽滞带, 防闪断)
        labels = self.data.labels
        active_indices=(list(range(len(labels))) if search_classes is None else sorted(search_classes))
        # 各类是否'白/灰类'(极低饱和占比 >0.6)：这类颜色模型太泛(白墙/纸/
        # 皮肤高光/眼镜反光都是'亮+低饱和'), 需要比其它类更严的判据。
        whites = []
        for i in range(len(labels)):
            h = self.data.class_hist(i)
            if h is None:
                whites.append(False)
                continue
            t3 = np.asarray(h, np.float32).reshape(8, 8, 8)
            t3 = t3 / max(float(t3.sum()), 1e-9)
            whites.append(float(t3.sum(axis=(0, 2))[0]) > 0.60)

        whites = self._bright_localizer_flags()
        short_side = min(self.W, self.H)    # 跟踪期尺寸地板 / 阶段2扫描窗口共用
        fracs = self.data.class_scan_fracs()

        # ---- 阶段 1：逐目标跟踪（每帧执行，约 11 窗/目标） ----
        if not search_only:
            self._prefetch_generic_tracks()
        for tgt in ([] if search_only else self.scan_targets[:]):
            tgt["verified"] = False
            if whites[tgt["cls"]]:
                self._track_localized_white(tgt, th_keep)
                continue
            if self._generic_class(tgt['cls']):
                self._track_generic_target(tgt)
                continue
            if tgt.get('colour_boundary'):
                self._track_colour_target(tgt)
                continue
            # 上一帧没验住(fail/cfail>0)说明物体在快速移动或被遮挡 -> 启用
            # 更远的搜索圈把它抓回来; 正常跟随时不加窗口, 保住帧率。
            # wide 触发: 上一帧没验住, 或者上一帧的位移已经顶到 ±0.25 搜索圈
            # 边缘(说明物体比搜索圈跑得快)。平时偏移圈只有 ±0.25, 追赶全靠
            # 这一圈, 所以这个 lag 信号很重要。
            wins, centers = self._refine_boxes(
                tgt["box"], tgt["asize"], self.data.class_ar(tgt["cls"]),
                wide=(tgt["fail"] > 0 or int(tgt.get("cfail", 0)) > 0
                      or int(tgt.get("lag", 0)) > 0))
            crops = np.stack([crop_resize(self.frame, b, self.in_size) for b in wins])
            feats = self.backbone(preprocess_batch(crops))
            coss = self.clf.cosine_batch(feats, self.data)          # (N,C) 余弦
            # 关键: 先对**所有**窗口跑完整门控链(背景门/负样本门/颜色门), 再挑
            # 窗口——位置必须从'过了颜色门'的窗口里选。原先门控只作用于存活
            # 判定, 位置却按**未门控**分数选, 于是某块颜色完全不符的区域(比如
            # 红色柜子)只要特征相似度高就能把框拽过去(用户反馈的'明明颜色不同,
            # 框还是飘到红柜子上')。现在颜色错的窗口根本进不了位置候选。
            gg = self._bg_gate(feats, coss.copy())                  # 背景门(全窗口)
            gg = self._neg_gate(feats, gg)                          # 负样本门(全窗口)
            g = self._hist_gate(crops, gg.copy())                   # 颜色门(全窗口)
            own_col = g[:, tgt["cls"]]                              # 自身类(已门控)
            ok = own_col > 0
            if ok.any():
                idx = np.flatnonzero(ok)
                i_best = int(idx[own_col[idx].argmax()])
            else:
                # 所有窗口都被颜色门掩蔽(翻转/暗光导致颜色漂移): 退回未门控分数
                # 保持位置连续性, 同时让下面的 cfail 宽限计数生效
                own_col = coss[:, tgt["cls"]]
                i_best = int(own_col.argmax())
            if centers:
                # 位置连续性优先:
                #   当前位置(中心窗口)的自身类分数只要过保持门槛, 就**留在原地**
                #   ——跟随由下面的'亚步长加权重心'提供, 不靠'跳到分数更高的邻居
                #   窗口'。这样旁边衣服/桌面某一块分数略高, 也不会把框拽过去
                #   (用户反馈的'跟踪乱跑/认到衣服'主要就是这么来的)。
                #   只有中心确实不行(被遮挡/移动太快, 分数掉了)才允许跳到明显
                #   更好的窗口(要高 0.10), 保证快速移动时还能追上。
                i_ctr = max(centers, key=lambda j: float(own_col[j]))
                if own_col[i_ctr] >= th_keep:
                    i = i_ctr
                elif own_col[i_best] >= own_col[i_ctr] + 0.10:
                    i = i_best
                else:
                    i = i_ctr
            else:
                i = i_best
            # 肤色门在跟踪期同样生效: 选中的窗口若大部分是皮肤(脸/脖子/手),
            # 就不是目标。原先肤色门只在'新锁定'时生效, 目标一旦锁定再飘到
            # 脸上/脖子上就没人拦(截图里绿框压在脸上的原因)。
            # 判定线 0.55→0.65: 运行日志里反复出现 `lost '2' (skin)` —— 手拿
            # 涂改带时窗口(宽扁条)会把手一起框进来, 肤色占比超过 0.55 就把真
            # 目标释放了。而且实测误检的脸/眼镜窗口肤色只有 0.19, 本来就**不是
            # 靠这条拦住的**——它当前唯一的实际效果就是误杀真物体, 所以放宽。
            if not self._generic_class(tgt['cls']) and self._skin_ratio(wins[i]) > 0.65:
                tgt["fail"] = int(tgt.get("fail", 0)) + 1
                tgt["cfail"] = int(tgt.get("cfail", 0)) + 1
                if tgt["fail"] > (8 if tgt["pend"] >= 2 else 1):
                    if whites[tgt["cls"]]:
                        self._auto_negative(wins[i])   # 反复的误检 -> 自动负样本
                    self._scan_log(f"lost '{tgt['label']}' (skin)")
                    self.scan_targets.remove(tgt)
                continue
            # 【跟踪期"周围没有手"校验】(只对白/灰类, 本轮新增)
            #   这个演示学的是**手里举着**的物体——真目标框外圈必然有手(皮肤);
            #   而"锁定后飘到衣服/胸口/背景上"的那一块, 四周没有手。
            #   旧版仅在新锁定(放置段)时查这一条, 跟踪期**不查** -> 目标一旦从
            #   涂改带滑到同样又白又平的衣领上, 就永远钉住、全局搜索又被 skip_cls
            #   跳过, 于是"涂改带就在镜头前却回不来"(用户实测反馈)。
            #   这里连续多帧都没手 -> **释放**该目标, 让全局搜索重新去找真涂改带。
            #   只释放、不否决, 且阈值 0.02 比新锁定的 0.05 更宽松, 手只露一点
            #   也不会误释放。
            if whites[tgt["cls"]]:
                bb = wins[i]
                bw2 = float(max(1, bb[2] - bb[0]))
                bh2 = float(max(1, bb[3] - bb[1]))
                ex = (max(0, bb[0] - int(0.5 * bw2)), max(0, bb[1] - int(0.5 * bh2)),
                      min(self.W, bb[2] + int(0.5 * bw2)),
                      min(self.H, bb[3] + int(0.5 * bh2)))
                a_ex = float(max(1, (ex[2] - ex[0]) * (ex[3] - ex[1])))
                a_in = float(max(1, bw2 * bh2))
                ring_skin = max(0.0, (self._skin_ratio(ex) * a_ex
                                      - self._skin_ratio(bb) * a_in)
                                / max(1.0, a_ex - a_in))
                if ring_skin < 0.02:
                    tgt["nohand"] = int(tgt.get("nohand", 0)) + 1
                else:
                    tgt["nohand"] = 0
                if int(tgt.get("nohand", 0)) >= 15:      # 约 0.5s 一直没手
                    self._auto_negative(wins[i])         # 反复的误检 -> 自动负样本
                    self._scan_log(f"lost '{tgt['label']}' (周围没有手)")
                    self.scan_targets.remove(tgt)
                    continue
            # 存活判定: own 用'颜色门之前'的分数(给颜色漂移留宽限, 见 cfail),
            # 颜色门的作用是'持续否决'——连续多帧颜色不符才释放目标。
            row_bg = gg[i]
            row = g[i]
            valid = row[row > 0]
            cls = int(row.argmax()) if valid.size else -1
            cos = float(valid.max()) if valid.size else -1.0
            srt = np.sort(valid)[::-1]
            amb = valid.size >= 2 and (srt[0] - srt[1]) < 0.04    # 类别歧义: 拒绝猜
            confirmed = tgt["pend"] >= 2
            own = float(row_bg[tgt["cls"]])       # 自身类: 颜色门之前的相似度
            # 目标'自身类'是否通过颜色门(必须看自身类, 不能看'有没有别的类通过')
            color_ok = float(row[tgt["cls"]]) > 0
            # 颜色合格计数改成"滞回"(过一帧减一, 不合格加一), 不再一次清零:
            # 原来只要偶尔有一帧颜色侥幸通过就清零 -> 残留的误检目标永远不释放
            # (实测: 诊断行显示颜色门 2:67/-100/-100 全被掩蔽, 框却一直挂在
            #  书架/背景上)。滞回后"大部分帧都不合格"的目标会稳定累计并释放。
            if color_ok:
                tgt["cfail"] = max(0, int(tgt.get("cfail", 0)) - 1)
            else:
                tgt["cfail"] = int(tgt.get("cfail", 0)) + 1
            # 颜色宽限 8 帧(约 0.4s): 物体翻转/换角度/手影遮挡会让颜色连续多帧
            # 判'不像', 宽限太短会把真目标误释放(用户反馈的'有时候会失去目标')。
            # 放心放宽的理由: 位置现在只从'过颜色门'的窗口里选, 误检目标根本
            # 不会被建立起来, 所以不需要靠快速释放来清误检。
            # 保持门槛不高抬(上一版抬到 0.68 是错的): 实测同一个白色涂改带
            # 换个角度只有 51%, 和背景块 52~65% 完全重叠——分数层面本来就分不开。
            # 真正把背景挡住的是**颜色门**(实测背景窗口整类被掩蔽 -100, 而真
            # 涂改带能过门拿 51)。所以这里回到宽松保持门槛, 避免误杀真物体。
            keep_th = th_keep
            keep = (own >= keep_th and tgt["cfail"] <= 8) if confirmed else \
                   (cls >= 0 and not amb and cos >= th_keep)
            if keep:
                tgt["verified"] = bool(color_ok and not amb)
                tgt["fail"] = 0
                if tgt["pend"] < 4:
                    # 确认计数上限 4(原来 2): 目标要连续 4 帧(~0.2s)稳定跟住
                    # 才会被画出来。瞬时的误匹配(眼镜反光、衣服某块偶然撞上)
                    # 一般撑不过 4 帧, 于是**根本不会被画出来**——治'某一帧
                    # 突然锁到眼镜框旁边'。真物体位置稳定, 0.2s 内即转正,
                    # 感知上几乎无差别。
                    if self._near(wins[i], tgt["box"], rel=0.75):
                        tgt["pend"] += 1
                    else:
                        tgt["box"] = wins[i]
                        tgt["ema"] = None
                        tgt["asize"] = (wins[i][2] - wins[i][0],
                                        wins[i][3] - wins[i][1])
                        tgt["cos"] = max(own, 0.0)
                        # 颜色门后的自身分数: -1 表示该窗口被颜色门掩蔽。
                        # 全局搜索用它判断该目标是否"位置存疑"(见 reloc_cls)。
                        tgt["gcos"] = float(row[tgt["cls"]])
                        continue
                # 位置跟随最佳窗口, 尺寸锚定 asize——窗口在 4 个尺度档间
                # 跳动是'框大小一直变'的根源, 这里只用最佳窗口的'位置',
                # 尺寸保持锚定值不动。
                nw, nh = wins[i][2] - wins[i][0], wins[i][3] - wins[i][1]
                aw, ah = tgt["asize"] if tgt["asize"] else (nw, nh)
                # 尺寸自适应: 只有当更优的尺寸档'明显更好'(相似度高出 0.06)
                # 才缓慢把锚定尺寸往那边挪(半步), 且尺寸也过 EMA, 防抖动。
                # 搜索档位只有 -12%/+15%，旧的 >20% 条件使正常尺寸
                # 更新永远不触发。新尺寸须优于当前尺寸的中心窗口。
                anchor_centers = [j for j in centers
                                  if wins[j][2] - wins[j][0] == aw
                                  and wins[j][3] - wins[j][1] == ah]
                anchor_score = max((float(own_col[j]) for j in anchor_centers),
                                   default=float(own_col[i]))
                if abs(min(nw, nh) - min(aw, ah)) > 0.08 * min(aw, ah) \
                        and float(own_col[i]) >= anchor_score + 0.02:
                    aw = int(0.5 * aw + 0.5 * nw)
                    ah = int(0.5 * ah + 0.5 * nh)
                    tgt["asize"] = (aw, ah)
                # 【跟踪尺寸地板 --track-size-floor(默认 0.35)】
                #   上面的尺寸自适应允许"更小的窗口若明显更好(+0.06)就缓慢跟过去",
                #   但**没有下限** —— 实测(2026-10-13 diag 三帧): 白类锚定尺寸被
                #   一路缩到 62×24, 只有注册长边(0.52×600=312)的 20%, 最后变成
                #   画面底部白衬衫上的一条细条, 而真物体完全不被检出。
                #   低于"该类注册长边 × 此比例"时不再跟随更小的窗口, 并把已经
                #   缩小的目标抬回地板(自愈)。设 0 关闭。
                _fl = float(getattr(self.args, "track_size_floor", 0.35) or 0.0)
                if _fl > 0.0:
                    _fs = max(16, int(_fl * fracs[tgt["cls"]] * short_side))
                    # fracs 记录的是注册长边；下限也必须约束长边。
                    if max(aw, ah) < _fs:
                        _k = _fs / float(max(1, max(aw, ah)))
                        aw, ah = int(aw * _k), int(ah * _k)
                        tgt["asize"] = (aw, ah)
                # 亚步长中心: 框中心不再直接取'最佳窗口'的离散格点, 而是用最佳
                # 窗口及其邻域窗口的分数做加权重心。窗口偏移是离散的(0/±0.12/
                # ±0.25/±0.45), 直接取格点会让框在移动时一格一格地跳(用户反馈的
                # '容易乱跳'); 加权重心给出连续位置, 跟随更平滑、也更跟手——
                # 代价只是几次乘加, 不影响帧率。
                ccx = ccy = wsum = 0.0
                base = float(own_col[i])
                for j2 in range(len(wins)):
                    wgt = float(own_col[j2]) - (base - 0.06)
                    # 邻域半径 0.35→0.50: 位置现在主要靠重心插值来跟, 半径放宽
                    # 让 ±0.45 外圈邻居也参与加权, 跟随幅度和原来相当,
                    # 不会因为'不再跳窗口'而变拖。
                    if wgt <= 0 or not self._near(wins[j2], wins[i], rel=0.50):
                        continue
                    ccx += wgt * (wins[j2][0] + wins[j2][2])
                    ccy += wgt * (wins[j2][1] + wins[j2][3])
                    wsum += wgt
                if wsum > 0:
                    cxm, cym = int(round(ccx / wsum / 2.0)), int(round(ccy / wsum / 2.0))
                else:
                    cxm = (wins[i][0] + wins[i][2]) // 2
                    cym = (wins[i][1] + wins[i][3]) // 2
                box_det = (cxm - aw // 2, cym - ah // 2, cxm + aw // 2, cym + ah // 2)
                ob = tgt["box"]
                tgt["ema"], nb = self._ema_box(box_det, tgt["ema"])
                tgt["box"] = nb
                # 位移是否已顶到 ±0.25 搜索圈边缘(物体跑得比圈快) -> 下一帧
                # 启用 wide 追赶圈。这是'偏移圈只有 ±0.25'之后保证跟得上的关键。
                mv = (abs((nb[0] + nb[2]) - (ob[0] + ob[2]))
                      + abs((nb[1] + nb[3]) - (ob[1] + ob[3]))) * 0.5
                tgt["lag"] = 1 if mv >= 0.16 * max(16, min(aw, ah)) else 0
                # 类别即时更正: 核验明确指向不同的类(非歧义、过保持门槛)才切换。
                # 但若那个类已经被另一个目标占着, 就不能切过去——否则同一个类
                # 会同时挂两个目标, 屏幕上出现两个同色框(截图里两个绿框)。
                if (cls >= 0 and not amb and cls != tgt["cls"] and cos >= th_keep
                        and not any(t2 is not tgt and t2["cls"] == cls
                                    for t2 in self.scan_targets)):
                    old_lab = tgt["label"]
                    tgt["cls"], tgt["label"], tgt["cos"] = cls, labels[cls], cos
                    tgt["gcos"] = cos
                    self._scan_log(f"relabel '{old_lab}' -> '{labels[cls]}' (cos={cos:.2f})")
                else:
                    tgt["cos"] = max(own, 0.0)
                    tgt["gcos"] = float(row[tgt["cls"]])
            else:
                tgt["fail"] += 1
                # 已确认目标宽限 8 帧(约 0.4s)：物体快速移动/短暂遮挡时会连续几帧
                # 找不到匹配——宽限期内会用更远的搜索圈(wide)重新抓回来；
                # 未确认目标 1 帧即弃(防幻影)。
                if tgt["fail"] > (8 if confirmed else 1):
                    if whites[tgt["cls"]] and not confirmed:
                        self._auto_negative(wins[i])   # 反复的未确认误检 -> 自动负样本
                    self._scan_log(f"lost '{tgt['label']}' (own={own:.2f} cos={cos:.2f} amb={amb})")
                    self.scan_targets.remove(tgt)

        # 重叠去重：不同类目标高度重叠/大框套小框多半是同一物体被两类
        # 同时咬住，保留相似度高的、去掉低的（同一物体不重复框）。
        # 用包含率(_same_obj)而非 IoU：大框罩小框时 IoU 远低于 0.5 会漏判。
        if len(self.scan_targets) >= 2:
            keep = []
            for t in sorted(self.scan_targets, key=lambda t: -t["cos"]):
                if any(k["cls"] != t["cls"] and _same_obj(t["box"], k["box"])
                       for k in keep):
                    continue
                keep.append(t)
            self.scan_targets = keep
        # 硬性保证: 同一个类只允许一个目标, 屏幕上一个类只画一个框。
        # (标签切换等异常路径可能造成同类重复 -> 两个同色框)
        if len(self.scan_targets) >= 2:
            best_by_cls = {}
            for t in sorted(self.scan_targets, key=lambda t: -t["cos"]):
                best_by_cls.setdefault(t["cls"], t)
            self.scan_targets = list(best_by_cls.values())

        # ---- 阶段 2：全局多目标搜索（节奏调度） ----
        healthy_classes = {t["cls"] for t in self.scan_targets
                           if t["pend"]>=4 and t.get("verified",False)
                           and t.get("fail",0)==0
                           and (t.get("boundary_age",0)<=6 or t.get("motion_quality",0.)>=.75)}
        if self._async_search is not None and not search_only:
            self._async_search_tick(healthy_classes,whites)
            self._update_scan_display()
            return
        if len(healthy_classes)==len(labels):
            # Every class already has current evidence. The previous code still
            # ran the entire coarse network batch before skipping those classes.
            self._update_scan_display()
            return
        recovering = any((t.get("ever_confirmed") or t["pend"] >= 4)
                          and t.get("fail", 0) > 0
                         for t in self.scan_targets)
        any_locked = any(t["pend"] >= 4 and t.get("verified", False)
                         for t in self.scan_targets)
        search_interval = 2 if recovering else (12 if any_locked else max(1, self.args.scan_interval))
        if self.frame_idx % search_interval != 0:
            self._update_scan_display()
            return
        side_cap = max(self.W,self.H) if any(self._generic_class(ci) for ci in active_indices) else int(0.85 * short_side)
        scale_sides, seen = [], set()
        for ci,f in enumerate(self.data.class_scan_fracs()):
            if ci not in active_indices:continue
            for s in self.args.scan_scales:
                side = int(f * s * short_side)
                if 16 <= side <= side_cap and side not in seen:
                    seen.add(side)
                    scale_sides.append(side)
        if not scale_sides:
            self._update_scan_display()
            return
        per_cap = max(12, self.args.max_scan_windows // len(scale_sides))
        boxes = []
        scan_ars = sorted({self.data.class_ar(ci) for ci in active_indices})
        for side in scale_sides:
            for ar in scan_ars:
                boxes += build_scan_boxes(self.W, self.H, side, per_cap, ar=ar)
        # 前景连通域候选：外接矩形天然就是物体在画面里的真实大小——贴近
        # 镜头的大物体也能被完整框住, 不受'注册尺度档'(0.45~3.0 倍)限制。
        fgm = (self._fg_mask(self.frame.astype(np.float32))
               if self.bg_model is not None else None)
        if fgm is not None:
            ncc, _, st, _ = cv2.connectedComponentsWithStats(fgm, 8)
            min_a = 0.002 * self.W * self.H
            added = []
            for i in range(1, ncc):
                if int(st[i, cv2.CC_STAT_AREA]) < min_a:
                    continue
                x = int(st[i, cv2.CC_STAT_LEFT])
                y = int(st[i, cv2.CC_STAT_TOP])
                w = int(st[i, cv2.CC_STAT_WIDTH])
                h = int(st[i, cv2.CC_STAT_HEIGHT])
                for ci in active_indices:
                    added.append(self._rect_box((x, y, x + w, y + h),
                                                self.data.class_ar(ci)))
            if added:
                boxes = list(dict.fromkeys(boxes + added))
        if not boxes:
            self._update_scan_display()
            return
        # 前景预过滤(性能核心)：先在背景减除得到的前景掩码上筛掉静态背景
        # 窗口, 只对剩下的窗口跑骨干前向——静态区域完全不进入特征计算
        fgv = self._fg_frac(boxes, fgm)
        if (fgv is not None and float(fgv.max()) > 0.25
                and int((fgv >= 0.03).sum()) >= 5):
            # 仅当'确有明显前景区域'时才启用过滤(性能优化); 否则回退不过滤
            # ——背景模型失真/物体被吸收进背景时, 过滤会把真物体一起剔除,
            # 表现为'物体就在眼前却全 0%'
            keep = fgv >= 0.03
            boxes = [b for b, k in zip(boxes, keep) if k]
            fgv = fgv[keep]
        else:
            fgv = None
        self._fgv = fgv
        crops = np.stack([crop_resize(self.frame, b, self.in_size) for b in boxes])
        feats = self.backbone(preprocess_batch(crops))
        coss = self.clf.cosine_batch(feats, self.data)              # (N,C) 余弦
        if search_classes is not None:
            coss[:,[ci for ci in range(len(labels)) if ci not in search_classes]]=-1.
        coarse_raw = coss.copy()
        raw = coss.max(axis=0).copy()                               # 门控前原始分(诊断)
        coss = self._hist_gate(crops, coss)                         # 颜色门控
        h_gated = coss.max(axis=0).copy()                           # 颜色门后(诊断)
        coss = self._bg_gate(feats, coss)                           # 背景负样本门控
        b_gated = coss.max(axis=0).copy()                           # 背景门后(诊断)
        coss = self._neg_gate(feats, coss)                          # 用户标记的负样本门控
        coss = self._shape_weight(boxes, coss)                      # 结构权重(80%)
        # 逐类全局最佳分数(未锁定也保留)
        self._global_cosv = np.maximum(coss.max(axis=0), 0.0).astype(np.float32)
        self._diag_stage = np.stack([raw, h_gated, b_gated])         # (3,C) 诊断链

        # 逐类取候选：纯 argmax 选各自最佳窗口。
        # 不做尺寸偏好——'挑接近注册尺寸的窗口'会让举近(比注册时大)的物体
        # 被注册尺寸的窗口硬套, 框只盖住物体局部, 位置/大小全错。
        # 巨窗防治交给后续两道闸: 精化按类中位收缩尺寸档 + 放置时大窗加严。
        # "位置存疑"的已确认目标: 颜色门后的自身分数没过新锁定线, 说明它虽然还
        # 挂着, 但未必落在物体真正的中心——白物体的白色区域往往不止一块(涂改带
        # 的白色头部旁边还有同样白的尾部小尖)。跟踪只会在框附近 ±0.25/±0.70 里
        # 挪, 一旦当初锁偏就永远挪不回来; 全局搜索原来又把已锁定的类整个跳过,
        # 于是框卡在错位置一动不动(用户反馈的'框一直在涂改带尾部')。这里让这类
        # 目标也参与全局搜索, 给它一次"重定位"的机会。
        reloc_cls = {}
        for t in self.scan_targets:
            if t["pend"] >= 2 and t["cls"] not in healthy_classes:
                reloc_cls[t["cls"]] = t
        cands = []
        rej = {}          # 每类候选被卡在哪一关(屏上诊断: 直接看出漏检原因)
        for ci in active_indices:
            # 无效小窗不能占据 argmax 后让整类退出，先过滤再选候选。
            valid_geom = np.array([self._geom_gate_ok(b, ci, fracs) for b in boxes])
            ranked = np.where(valid_geom, coss[:, ci], -1.0)
            j = int(ranked.argmax())
            cos = float(ranked[j])
            th_ci = th_lock
            # (这里原有一条'低纹理类 +0.12'的规则, 已删除——它和下面白/灰类
            #  的 -0.12 作用在同一个类上会相互抵消, 而且'挡背景'的职责现在
            #  由颜色门承担(白/灰类重合度门槛 0.55), 不再需要靠抬门槛。)
            if whites[ci]:
                # 白/灰类新锁定**抬高一档**(58→70): 这一类颜色模型太泛(白墙/
                # 白衣服/白背景都能蹭上), 实测背景块能拿到 52~65, 之前把门槛
                # 降到 46 是为了照顾斜角度的涂改带, 结果**背景也一起进来了**
                # (用户反馈的'之前同一个框不会误判, 现在会')。恢复到严格档:
                # 宁可斜角度/远距离漏一点, 也不要背景乱冒框。
                th_ci += 0.12
            if ci in reloc_cls:
                # 该类已有锁定目标、只是"位置存疑"。重定位不是新锁定: 目标肯定
                # 在手上, 只是位置可能偏了——用低一档的门槛去找它真正的位置,
                # 后面的放置闸门(周围有手/中心-周围对比/形状/肤色)继续兜底。
                th_ci = min(th_ci, th_keep + 0.05)
            if cos >= th_ci:
                fg = float(self._fgv[j]) if self._fgv is not None else 1.0
                cands.append({"cos": cos, "box": boxes[j], "ci": ci, "fg": fg})
            else:
                rej[ci] = f"分低{cos * 100:.0f}<{th_ci * 100:.0f}"
                # 粗框混入手/背景时颜色门可能拒绝，但应先尝试框内精化。
                # 原始高分仅用于安排搜索；精化后必须重过全部门控和锁定线。
                raw_ranked = np.where(valid_geom, coarse_raw[:, ci], -1.0)
                jr = int(raw_ranked.argmax())
                if float(raw_ranked[jr]) >= th_ci:
                    fg = float(self._fgv[jr]) if self._fgv is not None else 1.0
                    cands.append({"cos": float(coss[jr, ci]), "box": boxes[jr],
                                  "ci": ci, "fg": fg, "refine_th": th_ci})
        # 前置几何约束: 面积与该类注册尺寸不同量级的候选直接剔除
        keep = []
        for cd in cands:
            if self._geom_gate_ok(cd["box"], cd["ci"], fracs):
                keep.append(cd)
            else:
                rej[cd["ci"]] = "尺寸差太多"
        cands = keep
        # 已确认锁定的类跳过精化(网格搜索最耗算力, 跟踪每帧已在维护位置)
        # ——这是识别帧率的主要瓶颈, 锁定后即省下大部分计算。
        # 但"位置存疑"(reloc_cls)的锁定类不跳过: 正是要给它们重定位的机会。
        skip_cls = {t["cls"] for t in self.scan_targets
                    if t["pend"] >= 2 and t["cls"] not in reloc_cls}
        cands = [cd for cd in cands if cd["ci"] not in skip_cls]
        for ci in skip_cls:
            rej[ci] = "已锁定"

        # ---- 精细化：按类别参考尺寸收缩搜索，把窗口精确贴到物体上 ----
        # （尺寸锚定类别注册中位而非候选框本身：巨候选也能被收缩回紧框；
        #   偏移步长仍按候选框大小，保证大候选也能覆盖到物体真实位置）
        fine_cands = [cd for cd in cands if not whites[cd["ci"]]]
        if fine_cands:
            fb_all, owner = [], []
            for cd in fine_cands:
                ref = int(fracs[cd["ci"]] * short_side)
                for fb in self._fine_boxes(cd["box"], ref_side=ref,
                                           ar=self.data.class_ar(cd["ci"])):
                    fb_all.append(fb)
                    owner.append(cd["ci"])
            if fb_all:
                fcrops = np.stack([crop_resize(self.frame, b, self.in_size) for b in fb_all])
                ffeats = self.backbone(preprocess_batch(fcrops))
                fcoss = self.clf.cosine_batch(ffeats, self.data)
                fcoss = self._hist_gate(fcrops, fcoss)              # 颜色门控
                fcoss = self._bg_gate(ffeats, fcoss)                # 背景负样本门控
                fcoss = self._neg_gate(ffeats, fcoss)               # 负样本门控
                fcoss = self._shape_weight(fb_all, fcoss)           # 形状权重(与粗扫一致)
                best = {}
                for k, ci in enumerate(owner):
                    c = float(fcoss[k, ci])
                    if c > best.get(ci, (-1.0, -1))[0]:
                        best[ci] = (c, k)
                for cd in cands:
                    if cd["ci"] in best and best[cd["ci"]][0] > cd["cos"]:
                        k = best[cd["ci"]][1]
                        cd["cos"] = float(fcoss[k, cd["ci"]])
                        cd["box"] = fb_all[k]

        # 搜索窗只提供位置，白/灰类必须重新分类边界贴合后的图像。
        for ci, white in enumerate(whites):
            if not white or ci in skip_cls:
                continue
            seeds = [cd["box"] for cd in cands if cd["ci"] == ci]
            eligible = [j for j in np.argsort(coarse_raw[:, ci])[::-1]
                        if self._geom_gate_ok(boxes[j], ci, fracs)
                        and max(boxes[j][2]-boxes[j][0], boxes[j][3]-boxes[j][1])
                            >= .5*fracs[ci]*short_side
                        and coarse_raw[j, ci] >= th_lock][:3]
            seeds += [boxes[j] for j in eligible]
            fitted_boxes, occupancy_by_box = [], {}
            # A search must not run five GrabCuts before the next GUI frame.
            # Alternate full-scene proposals with one rotating local seed.
            # Every result is classified on this frame, never reused as a score.
            rounds = getattr(self,"_boundary_search_rounds",{})
            self._boundary_search_rounds = rounds
            turn = rounds.get(ci,0)
            rounds[ci] = turn+1
            seeds = list(dict.fromkeys(seeds))
            full_fits = []
            recovering_identity = any(t["cls"]==ci and (t.get("ever_confirmed") or t.get("pend",0)>0)
                                      for t in self.scan_targets)
            if recovering_identity or turn%2==0 or not seeds:
                full_fits = bright_component_candidates(self.frame, (0, 0, self.W, self.H),
                                                 self.data.class_ar(ci),max_candidates=8,
                                                 accept_seed=lambda box: self._bright_seed_has_structure(box,ci))
                full_fits=self._refine_bright_boundaries(full_fits,ci)
                if not recovering_identity:
                    seeds = []
            else:
                seeds = [seeds[(turn//2)%len(seeds)]]
            for fitted,occupancy in full_fits:
                if occupancy >= .45 and max(fitted[2]-fitted[0], fitted[3]-fitted[1]) >= .4*fracs[ci]*short_side:
                    fitted_boxes.append(fitted)
                    occupancy_by_box[fitted] = occupancy
                    # When the full-frame boundary proposal already verifies,
                    # additional local GrabCut calls cannot improve its identity.
                    full_score = float(self._score_localized_boxes([fitted])[0, ci])
                    if full_score >= th_lock and self._localized_geometry_ok(fitted, ci) \
                            and self._shape_gate_ok(fitted, ci, True) \
                            and self._cs_gate_ok(fitted, cs_th=.24):
                        seeds = []
            for seed in dict.fromkeys(seeds):
                # Coarse grid alignment can cut the object; give boundary
                # fitting context without treating that context as foreground.
                sw, sh = seed[2]-seed[0], seed[3]-seed[1]
                search = (max(0, seed[0]-int(.25*sw)), max(0, seed[1]-int(.25*sh)),
                          min(self.W, seed[2]+int(.25*sw)), min(self.H, seed[3]+int(.25*sh)))
                results = bright_component_candidates(self.frame, search, self.data.class_ar(ci),
                                              max_candidates=6,
                                              accept_seed=lambda box: self._bright_seed_has_structure(box,ci))
                results=self._refine_bright_boundaries(results,ci)
                for fitted,occupancy in results:
                    if occupancy >= .45 and fitted not in occupancy_by_box \
                            and max(fitted[2]-fitted[0], fitted[3]-fitted[1]) >= .4*fracs[ci]*short_side:
                        fitted_boxes.append(fitted)
                        occupancy_by_box[fitted] = occupancy
            cands = [cd for cd in cands if cd["ci"] != ci]
            if not fitted_boxes:
                rej[ci] = "未找到完整物体边界"
                continue
            fitted_scores = self._score_localized_boxes(fitted_boxes)[:, ci]
            evidence = self._last_localized_evidence
            best_index = int(np.argmax(fitted_scores))
            self._diag_localized[ci] = evidence[best_index][ci]
            found = False
            failure = self._diag_localized[ci]["reason"]
            for j in self._localized_candidate_order(fitted_boxes,fitted_scores,ci):
                fitted, score = fitted_boxes[j], float(fitted_scores[j])
                # A fitted crop uses the normal lock line, together with
                # independent boundary, contrast, color and background checks.
                if score < th_lock:
                    if failure == "通过":
                        failure = f"相似度{score:.0%}<{th_lock:.0%}"
                    continue
                if not self._localized_geometry_ok(fitted, ci):
                    failure = "主体长宽比与学习框不符"
                    continue
                if not self._shape_gate_ok(fitted, ci, True):
                    failure = "形状核验未通过"
                    continue
                if not self._cs_gate_ok(fitted, cs_th=.24):
                    failure = "与周围太像"
                    continue
                existing=next((t for t in self.scan_targets if t["cls"]==ci),None)
                new_identity=existing is None or not existing.get("ever_confirmed") \
                             or not self._near(fitted,existing["box"],rel=.75)
                if new_identity and not self._new_white_identity_ok(fitted,ci,score):
                    failure="注册结构外观不符"
                    continue
                if score >= th_lock:
                    cands.append(dict(box=fitted, ci=ci, cos=score, fg=1.,
                                      boundary_ok=True,
                                      boundary_occupancy=occupancy_by_box[fitted]))
                    found = True
                    self._diag_localized[ci] = evidence[j][ci]
                    break
            if not found:
                rej[ci] = "贴合拒绝·" + failure

        # ---- 候选生效：同步已有同类目标 / 孵化新目标 ----
        placed = []
        for cd in sorted(cands, key=lambda d: -d["cos"]):
            b, ci, cos = cd["box"], cd["ci"], cd["cos"]
            if "refine_th" in cd and cos < cd["refine_th"]:
                rej[ci] = f"精化未通过{cos * 100:.0f}<{cd['refine_th'] * 100:.0f}"
                continue
            side = max(b[2] - b[0], b[3] - b[1])
            ref_side = int(fracs[ci] * short_side)         # 该类注册框中位边长
            if side > ref_side * 2.3 and not cd.get("boundary_ok"):
                rej[ci] = f"巨窗{side}>{int(ref_side * 2.3)}"
                continue        # 远超类注册尺寸的巨窗直接不孵化(巨框根源);
                                # 2.3 倍以内放行, 覆盖扫描的 2.2 档(举近的物体)
            # 人脸几何硬过滤：宽高比 1:1.2~1:1.5 (w/h≈0.65~0.85) 且检测框面积
            # 占比 > 15% —— 人脸/头部的典型几何, 直接判为误检, 不进入输出
            bw_ = float(max(1, b[2] - b[0]))
            bh_ = float(max(1, b[3] - b[1]))
            if (bw_ * bh_) / float(self.W * self.H) > 0.20 and 0.60 <= bw_ / bh_ <= 0.90:
                rej[ci] = "像人脸"
                continue
            # 皮肤/人脸区域前置排除：窗口大部分是肤色(脸/下巴/手) -> 非目标。
            # 判定线 0.65(与跟踪期一致): 手拿物体时指头常占窗口 3~4 成甚至更多,
            # 门槛太低会把真物体一起拒掉(日志里的 lost '2' (skin))。
            # 【白/灰类再收紧一档: --white-win-skin-th, 默认 0.30】
            #   白物体最大的误检来源是'背景里的白衣服 / 人身上的白块'——这类
            #   窗口里往往掺着脸/脖子/手(实测 0.25~0.55), 而真白物体框内肤色
            #   很低(实测 0.02~0.07), 中间有明显空档。这条**只作用于新落锁**,
            #   不碰跟踪期(跟踪期仍用 0.65), 所以不会造成"锁定后总是丢失目标"。
            #   若真物体被误拒(手挡住了物体), 把 --white-win-skin-th 设 0 关闭。
            sk = self._skin_ratio(b)
            skin_th = (float(getattr(self.args, "white_win_skin_th", 0.30) or 0.0)
                       if whites[ci] and not cd.get("boundary_ok") else 0.65)
            if not self._generic_class(ci) and skin_th > 0.0 and sk > skin_th:
                rej[ci] = f"肤色{sk:.2f}>{skin_th:.2f}"
                continue
            # 已有独立边界证据的候选无需以手部作为物体存在依据。
            # "手在旁边"门控(只对白/灰类新锁定生效):
            #   这个演示学的是**手里举着**的物体——物体旁边必然有手(皮肤)。
            #   而白色物体最大的误检来源恰恰是'背景里的白衣服/白椅子/白纸',
            #   它们**周围没有手**。实测背景那件白衣服分数(70)甚至高过远距离
            #   的真涂改带, 靠分数永远分不开; 但"旁边有没有手"是干净利落的判据。
            #   只作用于新锁定(不碰跟踪), 且只对白/灰类, 影响面很小。
            if whites[ci] and not cd.get("boundary_ok", False):
                ex = (max(0, b[0] - bw_ // 2), max(0, b[1] - bh_ // 2),
                      min(self.W, b[2] + bw_ // 2), min(self.H, b[3] + bh_ // 2))
                a_ex = float(max(1, (ex[2] - ex[0]) * (ex[3] - ex[1])))
                a_in = float(max(1, (b[2] - b[0]) * (b[3] - b[1])))
                ring_skin = max(0.0, (self._skin_ratio(ex) * a_ex
                                      - self._skin_ratio(b) * a_in)
                                / max(1.0, a_ex - a_in))
                # 【阈值 0.05 -> 0.25】0.05 太松、等于没拦(背景白条旁边往往是手/
                #   脸, 也能量出 0.05+)。逐帧量过: **真涂改带框外圈肤色 0.32~0.42,
                #   而背景误检(衣袖/衣物/白墙)只有 0.09~0.20** -> 0.25 正好把两拨
                #   分开, 且给真目标留了余量。--ring-skin-th 可调, 设 0 关闭。
                ring_th = float(getattr(self.args, "ring_skin_th", 0.25) or 0.0)
                # 【近距离豁免 --closeup-ring-skip(默认 0.35)】
                #   "外圈有手"这个判据只在**物体举得远、框小、手露在旁边**时成立。
                #   把物体举到镜头前时, 框会占满画面, 外圈是物体自己的边缘或另一只
                #   手, ring_skin 必然很低 —— 判据在近距离**结构性失效**。
                #   实测(2026-10-13 录屏 f00240): 特/色/背三关全满(87/87/87)的真
                #   涂改带, 因 ring_skin=0.09 被拒 -> 用户看到的"只能认出一瞬间"。
                #   所以候选框长边已 >= 该比例×画面短边时, 跳过这条门(仍保留颜色/
                #   背景/形状/中心-周围等所有其它门)。设 0 关闭豁免。
                skip_ratio = float(getattr(self.args, "closeup_ring_skip", 0.35) or 0.0)
                closeup = skip_ratio > 0.0 and side >= skip_ratio * short_side
                if ring_th > 0.0 and not closeup and ring_skin < ring_th:
                    rej[ci] = f"周围没有手({ring_skin:.2f}<{ring_th:.2f})"
                    continue
            # 分级置信度门槛(前景占比来自背景减除)：整体下调一档, 因为上面
            # 的结构权重已从'最多扣 85%'收到'最多扣 15%', 分数不再被压扁,
            # 门槛要同步下调才不会把真物体挡在外面(实测: 书 66 分被旧门槛
            # 62/72 卡住, 明明在眼前却识别不出)。
            fg = float(cd.get("fg", 1.0))
            area_ratio = float((b[2] - b[0]) * (b[3] - b[1])) / float(self.W * self.H)
            if fg < 0.15:
                base_th = 0.72          # 前景几乎为零(完全静态): 提高门槛
            elif area_ratio < 0.05:
                base_th = 0.52          # 前景内小目标(涂改带/耳机): 放宽防漏检
            else:
                base_th = 0.58          # 前景内一般目标(保证真目标稳定检出)
            edges = sum((b[0] <= 1, b[1] <= 1, b[2] >= self.W - 1, b[3] >= self.H - 1))
            th_eff = base_th + (0.05 if edges >= 2 else 0.0)
            if side > ref_side * 1.6 and not cd.get("boundary_ok"): # 未贴合的大窗
                th_eff += 0.05                             # 背景稀释)才加严; 原来
                                                           # 1.3 倍就 +0.10, 把'举近
                                                           # 物体'直接卡在门外
            if whites[ci] and not cd.get("boundary_ok", False):
                th_eff += 0.06    # 白/灰类放置门槛也同步加严(同上: 优先不误检)
            # 重定位放行: 该类已有锁定目标且位置存疑, 这个候选又被全局搜索/
            # 精化一路筛到"比现位置明显更好", 就允许它用更低的门槛落地——否则
            # 高门槛会把唯一能纠正位置的候选也挡掉, 框就永远回不来了。
            rt = reloc_cls.get(ci)
            if cos < th_eff and not (rt is not None
                                     and cos >= float(rt.get("gcos", -1.0)) + 0.08):
                rej[ci] = f"分低{cos * 100:.0f}<{th_eff * 100:.0f}"
                continue
            # 中心-周围对比门控：窗口与其外圈背景的颜色分布几乎一样时拒绝
            # 锁定——纯色背景/桌面/墙面冒充物体时(相似度可能很高)靠这道闸拦住。
            # 白/灰类门槛更严(0.16→0.24): 这类颜色到处都是, 衣服/墙/桌面这些
            # '一整片均匀区域'如果落在边界上(外圈掺进了别的东西), 原来 0.16 的
            # 线就放它过去了(实测'锁到后面人的灰衣服上')。要求更明显的反差。
            if not self._cs_gate_ok(b, cs_th=(0.24 if whites[ci] else 0.16)):
                rej[ci] = "与周围太像"
                continue
            if not self._shape_gate_ok(b, ci, whites[ci]):   # 形状/轮廓门控(弱化颜色)
                rej[ci] = "形状不符"
                continue
            t_same = next((t for t in self.scan_targets if t["cls"] == ci), None)
            if t_same is not None:
                if cd.get("boundary_ok"):
                    if t_same.get("ever_confirmed") and int(t_same.get("fail", 0)) <= 4:
                        old = t_same["box"]
                        size_ratio = ((b[2]-b[0])*(b[3]-b[1])) / max(1.,(old[2]-old[0])*(old[3]-old[1]))
                        good_motion = t_same.get("motion_quality",0.) >= .75
                        size_ok = .8 <= size_ratio <= 1.25 if good_motion else .65 <= size_ratio <= 1.6
                        if not self._localized_boundary_consistent(b,t_same,cos):
                            rej[ci] = "候选边界与已跟踪主体不一致"
                            continue
                    if not self._near(b, t_same["box"], rel=.75):
                        # A distant proposal is a new identity confirmation,
                        # not an instant transfer of the old confirmed status.
                        t_same.update(box=b, asize=(b[2]-b[0], b[3]-b[1]), ema=None,
                                      pend=0, ever_confirmed=False,
                                       last_evidence_frame=-1, velocity=(0.,0.),
                                       appearance_bank=[])
                    self._accept_localized_target(t_same, b, cos, "全局重新核验", True,
                                                  occupancy=cd.get("boundary_occupancy"))
                    rej[ci] = f"全局贴合确认{cos:.0%}"
                    continue
                if t_same["pend"] < 2:
                    # 未确认: 全局搜索每轮命中的位置必须与上一轮相近才累计确认
                    # ——原先不分位置每轮都 +1, 于是'背景某处冒一次、另一处又冒
                    # 一次'也会被确认画框(随机乱框/闪烁的根源)。位置一变就重置
                    # 计数, 只有连续多轮稳定出现在同一处的才是真目标。
                    if not self._near(b, t_same["box"], rel=0.75):
                        rb = b if cd.get("boundary_ok") else self._rect_box(b, self.data.class_ar(ci))
                        t_same.update(box=rb, asize=(rb[2] - rb[0], rb[3] - rb[1]),
                                      ema=None, cos=cos, gcos=cos, fail=0)
                        continue
                    rb = b if cd.get("boundary_ok") else self._rect_box(b, self.data.class_ar(ci))
                    t_same.update(box=rb, asize=(rb[2] - rb[0], rb[3] - rb[1]),
                                  ema=None, cos=cos, gcos=cos, fail=0)
                    t_same["pend"] = min(4, t_same["pend"] + 1)
                else:
                    # 已确认: 位置/尺寸交给逐帧跟踪维护(平滑), 全局搜索只确认
                    # '它还在'。原先这里靠 _iou<0.2 硬替换框, 每 8 帧就把平滑
                    # 好的框重置成搜索窗口——'框忽然变一下/大小跳'的来源之一。
                    # 例外: 该类的锁定"位置存疑", 而全局搜索这次在**别处**找到
                    # 了一个可信(过新锁定线)的同类窗口 -> 说明当初锁偏了, 把框
                    # 挪到这个更好的位置(用户反馈的'框一直钉在涂改带尾部')。
                    if (ci in reloc_cls and cos >= th_lock
                            and not self._near(b, t_same["box"], rel=0.75)):
                        rb = b if cd.get("boundary_ok") else self._rect_box(b, self.data.class_ar(ci))
                        t_same.update(box=rb, asize=(rb[2] - rb[0], rb[3] - rb[1]),
                                      ema=None, cos=cos, gcos=cos, fail=0, cfail=0)
                        self._scan_log(
                            f"relocate '{labels[ci]}' cos={cos:.2f} box={rb}")
                        rej[ci] = f"重定位{cos * 100:.0f}"
                        continue
                    t_same["fail"] = 0
                    t_same["pend"] = min(4, t_same["pend"] + 1)
                rej[ci] = f"跟踪中{cos * 100:.0f}"
                continue
            # 新目标孵化：与其它类目标/已放置候选高度重叠/包含 -> 同一物体不重复框
            if any(t["cls"] != ci and _same_obj(b, t["box"]) for t in self.scan_targets):
                rej[ci] = "与他框重叠"
                continue
            if any(o["ci"] != ci and _same_obj(b, o["box"]) for o in placed):
                rej[ci] = "与他框重叠"
                continue
            b = b if cd.get("boundary_ok") else self._rect_box(b, self.data.class_ar(ci))
            self.scan_targets.append({"box": b, "cls": ci, "label": labels[ci],
                                      "cos": cos, "gcos": cos,
                                      "asize": (b[2] - b[0], b[3] - b[1]),
                                      "ema": None, "fail": 0, "pend": 1,
                                      "verified": True,
                                      "coasting": False, "boundary_age": 0,
                                      "ever_confirmed": False,
                                      "last_evidence_frame": self.frame_idx,
                                      "boundary_occupancy": cd.get("boundary_occupancy")})
            if cd.get("boundary_ok"):
                self._init_target_motion(self.scan_targets[-1])
            placed.append(cd)
            rej[ci] = f"新锁定{cos * 100:.0f}"
            # 日志带上 skin=: 真物体与误检的"框内肤色"实测差很大, 打出来才能
            # 按真实数据微调 --white-win-skin-th, 不用猜。
            self._scan_log(f"lock '{labels[ci]}' cos={cos:.2f} skin={sk:.2f} box={b}")

        self._diag_reject = rej        # 供屏上诊断行显示'每类卡在哪一关'
        self._update_scan_display()

    def _update_scan_display(self):
        """从多目标列表派生显示状态：scan_result=最佳目标(中央大字),
        scan_cosv=逐类相似度条形图。
        只反映'已确认锁定(pend>=2)'的目标——不再把全局最佳匹配(可能是
        背景某块的 0.5x)混进来，否则'什么都没放却显示物体2 52%'会被
        误读成已识别。未锁定的类一律 0%。"""
        n = len(self.data.labels)
        vec = np.zeros(n, np.float32)
        conf = [t for t in self.scan_targets
                if t["pend"] >= 4 and t.get("verified", False)]
        if conf:
            best = max(conf, key=lambda t: t["cos"])
            self.scan_result = (best["box"], best["label"], best["cos"])
            for t in conf:
                if t["cos"] > vec[t["cls"]]:
                    vec[t["cls"]] = t["cos"]
        else:
            self.scan_result = None
        # 只显示'已锁定'目标的分数: 全局最佳(可能是背景 0.5x)混进来会让人
        # 误以为'没放物体也识别到了'(显示 54% 却没框)
        self.scan_cosv = self._smooth_cosv(self.scan_cosv, vec)
        self.scan_cosv[vec <= 0] = 0.0

    def _scan_log(self, msg, min_dt=1.0):
        now = time.time()
        if now - self._scan_log_t >= min_dt:
            print(f"[scan] {msg}")
            self._scan_log_t = now

    def _rect_box(self, b, ar):
        """把方形候选框按类注册框的宽高比(ar=w/h)改成矩形, 保持中心与长边。
        宽扁/竖长物体用方形窗口匹配时, 窗口含大片背景导致框偏、分数虚低。"""
        ls = max(b[2] - b[0], b[3] - b[1])
        if ar >= 1.0:
            rw, rh = ls, max(16, int(round(ls / ar)))
        else:
            rw, rh = max(16, int(round(ls * ar))), ls
        cx, cy = (b[0] + b[2]) // 2, (b[1] + b[3]) // 2
        x1 = int(np.clip(cx - rw // 2, 0, max(0, self.W - rw)))
        y1 = int(np.clip(cy - rh // 2, 0, max(0, self.H - rh)))
        return (x1, y1, x1 + rw, y1 + rh)

    def _refine_boxes(self, b, asize=None, ar=1.0, wide=False):
        """围绕锁定框生成局部细化窗口（3 种尺寸 × 17 个偏移, wide 再加 8 个）。
        尺寸以锚定尺寸 asize 为基准 ×{0.85, 1.0, 1.18}：1.18 放大档防物体
        靠近后框跟不上尺寸。
        偏移覆盖 ±0.45 锚定尺寸：远距离物体小时手部快速移动也能追上。
        wide=True 时额外加 ±0.70 与对角 ±0.55 一圈——只在目标上一两帧没验住
        (fail/cfail>0, 说明物体在快速移动或被短暂遮挡)时启用, 把跑出原来的
        ±0.45 搜索圈的目标重新抓回来; 平时不加, 以免每帧多算窗口拖慢帧率。
        返回 (窗口列表, 中心窗口下标列表)。"""
        if asize:
            aw, ah = asize
        else:   # 无锚定尺寸: 按类注册框宽高比矩形化(长边为基准)
            ls = max(b[2] - b[0], b[3] - b[1])
            if ar >= 1.0:
                aw, ah = ls, max(16, int(round(ls / ar)))
            else:
                aw, ah = max(16, int(round(ls * ar))), ls
        cap = int(0.85 * min(self.W, self.H))   # 尺寸放大同样封顶, 防整屏大框
        cx, cy = (b[0] + b[2]) // 2, (b[1] + b[3]) // 2
        # 偏移/尺寸方案(帧率优化, 窗口数 27→11):
        #   实测每个窗口约 0.5ms(裁剪+缩放+骨干网络), 跟踪每帧都跑, 是识别模式
        #   帧率的主要开销。这里把每目标每帧的窗口压到 11 个:
        #     · 基准尺寸: 中心 + 轴向 ±0.25(5 个)——位置精度不靠密集格点,
        #       框中心用邻域分数加权重心插值, 0/0.25 之间能给出连续位置
        #     · 放大尺寸(1.15): 同样 5 个——物体拿近时框要能变大
        #     · 收缩尺寸(0.88): 只在中心测 1 个——只为把过大的框慢慢收回,
        #       不需要铺满偏移
        #   追丢时(wide)才补 ±0.70/对角±0.55 那圈追赶窗口(+8 个)。
        off5 = [(0, 0), (int(aw * 0.25), 0), (-int(aw * 0.25), 0),
                (0, int(ah * 0.25)), (0, -int(ah * 0.25))]
        wide_off = []
        if wide:
            wide_off = [(int(aw * 0.70), 0), (-int(aw * 0.70), 0),
                        (0, int(ah * 0.70)), (0, -int(ah * 0.70)),
                        (int(aw * 0.55), int(ah * 0.55)), (-int(aw * 0.55), -int(ah * 0.55)),
                        (int(aw * 0.55), -int(ah * 0.55)), (-int(aw * 0.55), int(ah * 0.55))]
        plan = [((aw, ah), off5 + wide_off),
                ((int(aw * 1.15), int(ah * 1.15)), off5),
                ((int(aw * 0.88), int(ah * 0.88)), [(0, 0)])]
        outs, centers = [], []
        for (sw, sh), offs in plan:
            sw, sh = min(sw, cap), min(sh, cap)
            if sw < 16 or sh < 16:
                continue
            for dx, dy in offs:
                x1 = int(np.clip(cx + dx - sw // 2, 0, self.W - sw))
                y1 = int(np.clip(cy + dy - sh // 2, 0, self.H - sh))
                if dx == 0 and dy == 0:
                    centers.append(len(outs))
                outs.append((x1, y1, x1 + sw, y1 + sh))
        return outs, centers

    def _fine_boxes(self, b, n=3, ref_side=None, ar=1.0):
        """在候选框附近生成精细网格窗口（粗搜后的二次对齐, 性能与精度折中）：
        n×n 密集偏移把窗口精确贴到物体上；偏移步长按候选框边长的 0.25。
        尺寸：给定 ref_side(类别注册框中位边长)时按其 {0.8,1.0,1.3,1.6}
        收缩——巨候选能被拉回紧框, 1.6 档覆盖举得比注册时近的物体；
        未给时按候选框 {0.8,1.0,1.2,1.4}。
        尺寸封顶 72% 短边（与全局搜索一致，整屏大窗不参与）。"""
        w, h = b[2] - b[0], b[3] - b[1]
        cx, cy = (b[0] + b[2]) // 2, (b[1] + b[3]) // 2
        fx, fy = max(1, int(w * 0.18)), max(1, int(h * 0.18))
        cap = int(0.85 * min(self.W, self.H))
        outs = []
        if ref_side:
            base = [(int(ref_side * r), int(ref_side * r))
                    for r in (0.4, 0.6, 0.8, 1.0, 1.3, 1.6)]
        else:
            base = [(int(w * r), int(h * r)) for r in (0.8, 1.0, 1.2, 1.4)]
        # 按类注册框的宽高比矩形化(ref_side 代表长边): 宽扁/竖长物体用正方形
        # 窗口会含大片背景、窗口对不准, 这是'匹配到却框偏'的主因
        sizes = []
        for sw, sh in base:
            if ar >= 1.0:
                sizes.append((sw, max(16, int(round(sw / ar)))))
            else:
                sizes.append((max(16, int(round(sh * ar))), sh))
        for sw, sh in sizes:
            if sw < 16 or sh < 16 or sw > cap or sh > cap:
                continue
            for dy in range(-(n // 2), n // 2 + 1):
                for dx in range(-(n // 2), n // 2 + 1):
                    x1 = int(np.clip(cx + dx * fx - sw // 2, 0, self.W - sw))
                    y1 = int(np.clip(cy + dy * fy - sh // 2, 0, self.H - sh))
                    outs.append((x1, y1, x1 + sw, y1 + sh))
        return outs

    def _ema_box(self, b, ema=None, alpha=0.78, alpha_s=0.28):
        """定位框 EMA 平滑（防抖动；位置 alpha 偏跟手, 移动跟得上）。
        位置和尺寸都平滑：尺寸 alpha_s 明显小于位置——窗口在不同尺度档间
        跳动时显示框的尺寸只缓慢变化, 不再逐帧忽大忽小（'框大小一直变'的
        主因就是尺寸原先直接取当帧窗口）。
        alpha 取 0.78 / alpha_s 0.28：0.85 太跟手会抖('框经常抖动'), 0.72
        又偏拖('跟随效果不好'), 取中间; 尺寸压到 0.28 抑制尺寸抖动。
        每个目标各自持有 ema 状态（多目标互不干扰）。返回 (ema, 平滑后的框)。"""
        w, h = b[2] - b[0], b[3] - b[1]
        if ema is None or len(ema) < 4:
            ema = [float(b[0]), float(b[1]), float(w), float(h)]
        else:
            ema = list(ema)
            ema[0] += alpha * (b[0] - ema[0])
            ema[1] += alpha * (b[1] - ema[1])
            ema[2] += alpha_s * (w - ema[2])
            ema[3] += alpha_s * (h - ema[3])
        w = max(16, int(round(ema[2])))
        h = max(16, int(round(ema[3])))
        x1 = int(np.clip(ema[0], 0, self.W - w))
        y1 = int(np.clip(ema[1], 0, self.H - h))
        return ema, (x1, y1, x1 + w, y1 + h)

    def _near(self, b1, b2, rel=0.6):
        """两个窗口中心距离是否小于 rel×窗口边长（连续确认判定用）"""
        c1 = ((b1[0] + b1[2]) * 0.5, (b1[1] + b1[3]) * 0.5)
        c2 = ((b2[0] + b2[2]) * 0.5, (b2[1] + b2[3]) * 0.5)
        r = rel * max(b2[2] - b2[0], b2[3] - b2[1], 16)
        return (c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2 <= r * r

    # ---------------- 绘制 ----------------
    def draw(self):
        fr = self.frame
        hb = int(0.075 * self.H)
        bband = int(0.095 * self.H)     # 底栏加高一档: 容纳新增的'卡在哪一关'诊断行
        cv2.rectangle(fr, (0, 0), (self.W, hb), HEAD_BG, -1)
        cv2.rectangle(fr, (0, self.H - bband), (self.W, self.H), HEAD_BG, -1)

        # 顶栏：模式 + 标题 + fps
        mode_t = self.t("mode_a") if self.mode == "A" else self.t("mode_b")
        mode_c = GREEN_BOX if self.mode == "A" else CLASS_COLORS[2]
        self.text(mode_t, int(10 * self.bs), int(4 * self.bs), int(21 * self.bs), mode_c)
        self.text(self.t("title"), int(150 * self.bs), int(4 * self.bs), int(15 * self.bs), (180, 180, 180))
        fps_t = f"{self.fps:5.1f} fps"
        self.text(fps_t, self.W - int(110 * self.bs), int(4 * self.bs), int(21 * self.bs), WHITE)
        if self._async_search is not None and self.mode=="B":
            self.text(f"搜索 {self.search_fps:.1f}/s",self.W-int(110*self.bs),int(28*self.bs),
                      int(12*self.bs),(180,180,180))

        # 状态行
        self.text(self.status, int(10 * self.bs), hb + int(6 * self.bs), int(19 * self.bs), YELLOW)

        # 右上角常显：当前采集帧数设置
        hud = self.t("collect_n") % self.args.shots_per_box
        hpx = int(17 * self.bs)
        self.text(hud, self.W - int(10 * self.bs) - self.R.text_width(hud, hpx),
                  hb + int(8 * self.bs), hpx, (185, 185, 185))

        # B 模式：未定位到物体时，中央大字显示当前最佳匹配（相似度低于门槛则显示'无匹配'）
        if (self.mode == "B" and self.manual_box is None and self.scan_result is None
                and self.data.has_data()):
            cosv = self.active_cosv()
            if cosv is not None:
                idx = int(cosv.argmax())
                cos = float(cosv[idx])
                if any(t.get("coasting") for t in self.scan_targets):
                    txt, col = self.t("motion_hold"), GRAY_BOX
                elif cos >= self.args.scan_cos_th:
                    txt, col = f"{self.data.labels[idx]}  {cos:.0%}", self.class_color(idx)
                else:
                    txt, col = self.t("no_match"), (140, 140, 140)
                px = int(34 * self.bs)
                tw = self.R.text_width(txt, px)
                self.text(txt, (self.W - tw) // 2, hb + int(30 * self.bs), px, col)

        # 已学类别列表（缩略图 + 标签 + 实时相似度条）
        if self.data.has_data():
            cosv = self.active_cosv()
            y0 = hb + int(38 * self.bs)
            self.text(self.t("reg_list"), int(10 * self.bs), y0, int(17 * self.bs), (200, 200, 200))
            ts = int(28 * self.bs)           # 缩略图边长
            tx = int(10 * self.bs) + ts + int(8 * self.bs)
            row_h = int(42 * self.bs)
            bar_w = int(120 * self.bs)
            bar_h = max(4, int(8 * self.bs))
            for i, lab in enumerate(self.data.labels[:8]):
                ty = y0 + int(24 * self.bs) + row_h * i
                c = self.class_color(i)
                thumb = self.data.thumbs[i] if i < len(self.data.thumbs) else None
                if thumb is not None and thumb.any():
                    fr[ty:ty + ts, int(10 * self.bs):int(10 * self.bs) + ts] = \
                        cv2.resize(thumb, (ts, ts), interpolation=cv2.INTER_AREA)
                    cv2.rectangle(fr, (int(10 * self.bs), ty),
                                  (int(10 * self.bs) + ts, ty + ts), c, 2)
                else:
                    cv2.rectangle(fr, (int(10 * self.bs), ty),
                                  (int(10 * self.bs) + ts, ty + ts), c, -1)
                if cosv is not None:
                    self.text(f"{lab}  {max(cosv[i], 0.0):.0%}", tx, ty, int(17 * self.bs), c)
                    cv2.rectangle(fr, (tx, ty + int(22 * self.bs)),
                                  (tx + bar_w, ty + int(22 * self.bs) + bar_h), (90, 90, 90), -1)
                    fw = int(bar_w * np.clip(cosv[i], 0.0, 1.0))
                    if fw > 0:
                        cv2.rectangle(fr, (tx, ty + int(22 * self.bs)),
                                      (tx + fw, ty + int(22 * self.bs) + bar_h), c, -1)
                else:
                    self.text(f"{lab} ({len(self.data.shots[i])}{self.t('fr')})",
                              tx, ty, int(17 * self.bs), c)
            if len(self.data.labels) > 8:
                self.text(f"+{len(self.data.labels) - 8}", int(10 * self.bs),
                          y0 + int(24 * self.bs) + row_h * 8, int(17 * self.bs), (200, 200, 200))

        # 拖拽中的框
        if self.drawing and self.box_start and self.box_cur:
            b = norm_box(self.box_start, self.box_cur, self.W, self.H)
            cv2.rectangle(fr, (b[0], b[1]), (b[2], b[3]), GREEN_BOX, 2)
            self.text(f"{b[2]-b[0]}x{b[3]-b[1]}", b[0], max(0, b[1] - int(26 * self.bs)),
                      int(17 * self.bs), GREEN_BOX)

        # A 模式：待命名/采集中的注册框
        if self.mode == "A" and self.interaction in ("label", "collect") and self.pending_box:
            b = self.pending_box
            cv2.rectangle(fr, (b[0], b[1]), (b[2], b[3]), GREEN_BOX, 2)
            if self.interaction == "collect":
                prog = f"{min(self.collect_n,self.args.shots_per_box)}/{self.args.shots_per_box} | {getattr(self.collect_capture,'accepted',0)} valid"
                self.text(prog, b[0], max(0, b[1] - int(26 * self.bs)), int(19 * self.bs), GREEN_BOX)

        # B 模式：手选识别框
        if self.mode == "B" and self.manual_box is not None:
            b = self.manual_box
            if self.data.has_data() and self.manual_cosv is not None:
                idx = int(self.manual_cosv.argmax())
                cos = float(self.manual_cosv[idx])
                if cos >= self.args.scan_cos_th:
                    c = self.class_color(idx)
                    txt = f"{self.data.labels[idx]} {cos:.0%}"
                else:
                    c, txt = GRAY_BOX, self.t("no_match")
            else:
                c, txt = GRAY_BOX, "?"
            cv2.rectangle(fr, (b[0], b[1]), (b[2], b[3]), c, 2)
            self.text(txt, b[0], max(0, b[1] - int(26 * self.bs)), int(20 * self.bs), c)

        # B 模式：自动定位结果（多目标：每个学习过的物体各自一个框；
        # 最佳目标粗框大字，其余正常粗细；未确认的带'?'提示）
        if self.mode == "B" and self.scan_targets:
            for t in self.scan_targets:
                if t.get("coasting") and not t.get("verified"):
                    b = t["box"]
                    cv2.rectangle(fr, (b[0], b[1]), (b[2], b[3]), GRAY_BOX, 2)
                    self.text(self.t("motion_hold"), b[0], max(0, b[1]-int(24*self.bs)),
                              int(18*self.bs), GRAY_BOX)
            # 只绘制已确认(pend>=4)的目标: 背景/相似区域误匹配会立刻孵化, 但
            # 它们的窗口位置每帧都在跳, 要连续 4 帧(~0.2s)稳定跟住才转正——
            # 期间不画框, 瞬时误检(眼镜反光/衣服偶然撞上)根本不会出现。
            # 真物体位置稳定, 0.2s 内即转正。
            shown = [t for t in sorted(self.scan_targets, key=lambda t: -t["cos"])
                     if t["pend"] >= 4 and t.get("verified", False)]
            for k, t in enumerate(shown):
                idx = self.data.labels.index(t["label"])
                c = self.class_color(idx)
                b = t["box"]
                cv2.rectangle(fr, (b[0], b[1]), (b[2], b[3]), c, 3 if k == 0 else 2)
                self.text(f"{t['label']} {t['cos']:.0%}", b[0],
                          max(0, b[1] - int(26 * self.bs)),
                          int((22 if k == 0 else 18) * self.bs), c)

        # 标签输入弹窗
        if self.interaction == "label":
            w, h = int(640 * self.bs), int(110 * self.bs)
            x0, y0 = self.W // 2 - w // 2, self.H // 2 - h // 2
            cv2.rectangle(fr, (x0, y0), (x0 + w, y0 + h), (35, 35, 35), -1)
            cv2.rectangle(fr, (x0, y0), (x0 + w, y0 + h), GREEN_BOX, 2)
            self.text(f"{self.t('prompt_label')}  {self.pending_label}|",
                      x0 + int(20 * self.bs), y0 + int(14 * self.bs), int(28 * self.bs), WHITE)
            self.text(self.t("prompt_hint"),
                      x0 + int(20 * self.bs), y0 + int(64 * self.bs), int(15 * self.bs), (180, 180, 180))

        # 诊断行：门控链各段分数 原始 -> 颜色门后 -> 背景门后
        #   原始低        = 特征不像(尺度/角度/注册问题)
        #   过颜色门后归零 = 颜色门砍的
        #   过背景门后归零 = 背景门砍的
        d = getattr(self, "_diag_stage", None)
        if d is not None and d.shape[1] == len(self.data.labels):
            parts = []
            for i, lab in enumerate(self.data.labels):
                parts.append(f"{lab}:{d[0, i] * 100:.0f}/{d[1, i] * 100:.0f}/{d[2, i] * 100:.0f}")
            self.text(f"[best 原始/色门/背门] {'  '.join(parts)}", int(10 * self.bs),
                      self.H - bband + int(25 * self.bs), int(15 * self.bs),
                      (150, 200, 150))
        # 诊断行2：每个类'卡在哪一关'——直接指明漏检/误检的原因
        # (低分/尺寸差太多/巨窗/像人脸/肤色/与周围太像/形状不符/与他框重叠/已锁定/跟踪中)
        rj = getattr(self, "_diag_reject", None)
        if rj:
            parts2 = [f"{lab}:{rj.get(i, '-')}"
                      for i, lab in enumerate(self.data.labels)]
            self.text(f"[卡在哪] {'  '.join(parts2)}", int(10 * self.bs),
                      self.H - bband + int(44 * self.bs), int(15 * self.bs),
                      (255, 205, 140))
        # 底部按键提示
        hint = self.t("hint_keys_a") if self.mode == "A" else self.t("hint_keys_b")
        self.text(hint, int(10 * self.bs), self.H - bband + int(8 * self.bs),
                  int(16 * self.bs), (220, 220, 220))

    def flush_texts(self):
        self.R.render(self.frame, self.text_ops)
        self.text_ops = []

    def _record_frame_time(self, duration, read_ms, tick_ms, draw_ms):
        """FPS is completed frames / elapsed time, including slow search frames."""
        self._frame_times.append(float(duration))
        self._frame_time_total += duration
        while len(self._frame_times)>1 and self._frame_time_total-self._frame_times[0]>=2.:
            self._frame_time_total -= self._frame_times.popleft()
        self.fps = len(self._frame_times)/max(self._frame_time_total,1e-9)
        self._perf_current = dict(read=round(read_ms,1), recognition=round(tick_ms,1),
                                  render=round(draw_ms,1),
                                  frame_p95=round(float(np.percentile(self._frame_times,95))*1000,1),
                                  frame_max=round(max(self._frame_times)*1000,1))
        self._perf_current.update({k:round(v,1) for k,v in getattr(self,'_generic_perf',{}).items()})
        if self._async_search is not None:
            self._perf_current.update(search_hz=round(self.search_fps,1),
                search_ms=round(self._search_last_ms,1),result_age=round(self._search_age_ms,1),
                search_kind=getattr(self,'_search_last_kind',None),
                dropped_old_searches=getattr(self,'_search_dropped_age',0))

    # ---------------- 主循环 ----------------
    def run(self, cap):
        runtime_policy=WindowsRealtimePolicy()
        try:
            print(f"[runtime] Windows HighQoS={runtime_policy.enabled}",flush=True)
            cap=LatestFrameCamera(cap)
            if not getattr(self.args,"sync_search",False):
                self._async_search=SingleFlightWorker(self._run_background_search)
                print("[runtime] latest camera frame / asynchronous search / per-frame tracking",flush=True)
            cv2.namedWindow(WIN_NAME, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(WIN_NAME, self.on_mouse)
            self.set_mode("A")  # 打印并设置初始状态
            fail = 0
            previous_frame_end = time.perf_counter()
            while not self.quit:
                t0 = time.perf_counter()
                ret, raw = cap.read()
                read_end = time.perf_counter()
                if not ret:
                    fail += 1
                    self.raw_frame = None
                    self._capture_time = None
                    self.frame[:] = 0
                    for target in self.scan_targets:
                        target["verified"]=False
                    self._update_scan_display()
                    if fail == 1:
                        # No recognition jobs run during a camera outage.
                        # Invalidate once per outage, rather than rebuilding
                        # the same empty search state on every failed read.
                        self._invalidate_search()
                    self.set_status(self.t("cam_err"))
                    if fail > 30:  # 尝试重开摄像头（内部会自动换后端）
                        closed=False
                        try:
                            closed=cap.release()
                        except Exception:
                            pass
                        time.sleep(0.3)
                        try:
                            if closed:
                                cap = LatestFrameCamera(init_camera(self.args))
                            else:
                                print("[camera] 等待旧读取线程释放设备，稍后再试",flush=True)
                        except Exception as e:
                            print(f"[camera] 重开失败: {e}，稍后再试")
                            time.sleep(1.0)
                        fail = 1   # 保留'曾失败'标记，恢复首帧显示"摄像头已恢复"
                else:
                    if fail > 0:                      # 从读取失败中恢复，清掉错误提示
                        self.set_status(self.t("cam_ok"))
                    fail = 0
                    if raw.shape[1] != self.W or raw.shape[0] != self.H:
                        raw = cv2.resize(raw, (self.W, self.H), interpolation=cv2.INTER_AREA)
                    self.frame[:] = raw
                    self.raw_frame = raw   # 引用(非拷贝): cap.read()/resize 每帧都返回新数组
                    self._capture_time=cap.capture_time

                # Windows waitKey(1) can wait for a whole system timer tick.
                k = (cv2.pollKey() if hasattr(cv2,"pollKey") else cv2.waitKey(1)) & 0xFF
                self.handle_key(k)
                try:
                    if cv2.getWindowProperty(WIN_NAME,cv2.WND_PROP_VISIBLE)<1:
                        self.quit=True
                except cv2.error:
                    self.quit=True
                if self.quit:
                    break
                tick_start = time.perf_counter()
                if ret:
                    self.tick()
                tick_end = time.perf_counter()
                self.draw()
                self.flush_texts()
                if self._diag_pending:      # 按过 d: 此刻 frame 已带框+诊断文字
                    self._diag_pending = False
                    self._save_diag()
                if self._diag_writer is not None:
                    saved = self._diag_writer.poll()
                    if saved is not None:
                        self.set_status(f"诊断保存失败: {saved['error']}" if saved['error'] else
                                        f"诊断已存到 {saved['directory']}（{saved['count']} 张图 + diag.txt）")
                cv2.imshow(WIN_NAME, self.frame)
                self.frame_idx += 1
                frame_end = time.perf_counter()
                self._record_frame_time(frame_end-previous_frame_end, (read_end-t0)*1000,
                                        (tick_end-tick_start)*1000, (frame_end-tick_end)*1000)
                self._perf_current.update(capture_age=round((frame_end-cap.capture_time)*1000,1) if ret else None,
                                          camera_fps=round(cap.source_fps,1),
                                          camera_properties=cap.properties.copy(),
                                          camera_error=getattr(cap,'last_error',None),
                                          events=round((tick_start-read_end)*1000,1),
                                          skipped_camera_frames=cap.dropped_frames)
                if (frame_end-previous_frame_end>=.080 and
                        frame_end>=getattr(self,'_spike_log_after',0.)):
                    self._spike_log_after=frame_end+.5
                    print(f"[frame-spike] time={time.strftime('%Y-%m-%d %H:%M:%S')} "
                          f"frame={self.frame_idx} elapsed_ms={(frame_end-previous_frame_end)*1000:.1f} "
                          f"targets={len(self.scan_targets)} ms={self._perf_current}",flush=True)
                previous_frame_end = frame_end
                # 每 5 秒把"每类卡在哪一关"(屏上 [卡在哪] 那行)同步打到日志 ——
                # 纯诊断, 不参与任何判定; 便于离线排查漏检/误检原因。
                if t0 - getattr(self, "_rej_log_t", 0.0) >= 5.0:
                    self._rej_log_t = t0
                    rj = getattr(self, "_diag_reject", None) or {}
                    parts = []
                    for k, v in rj.items():
                        ki = int(k)
                        lab = (self.data.labels[ki]
                               if 0 <= ki < len(self.data.labels) else str(k))
                        parts.append(f"{lab}:{v}")
                    print(f"[perf] fps={self.fps:.1f} frame={self.frame_idx} "
                          f"targets={len(self.scan_targets)} ms={self._perf_current}", flush=True)
                    if parts:
                        print("[rej] " + "  ".join(parts), flush=True)
        except KeyboardInterrupt:
            pass
        finally:
            cap.release()
            if self._async_search is not None:
                self._async_search.close()
            if self._diag_writer is not None:
                self._diag_writer.close()
            runtime_policy.close()
            cv2.destroyAllWindows()
        print(f"[exit] frames={self.frame_idx} classes={len(self.data.labels)}")


# ----------------------------------------------------------------------------
# 自检：无摄像头/无 GUI，验证 学习 -> 识别 -> 扫描 -> 存取 全流程
# ----------------------------------------------------------------------------
def run_self_test(args):
    print("=== PEFSL bbox_demo 自检 ===")
    backbone = load_backbone(args)
    data = LearnedData()
    clf = PefslClassifier(args.classifier_type, args.number_neiboors)
    size = args.resolution_input
    rng = np.random.default_rng(0)

    # 构造两个外观差异明显的"物体"：红底白横纹 / 绿底蓝棋盘（BGR）
    tex1 = np.zeros((96, 96, 3), np.uint8); tex1[:] = (40, 40, 180)
    tex1[::16] = (255, 255, 255)
    tex2 = np.zeros((96, 96, 3), np.uint8); tex2[:] = (40, 180, 40)
    yy, xx = np.mgrid[0:96, 0:96]
    tex2[(xx // 12 + yy // 12) % 2 == 0] = (255, 0, 0)
    frame = np.full((360, 480, 3), 128, np.uint8)
    frame[60:156, 60:156] = tex1       # 'objA' @ (60,60)
    frame[200:296, 300:396] = tex2    # 'objB' @ (300,200)

    def register(box, label):
        feats = []
        for _ in range(args.shots_per_box):
            j = rng.integers(-3, 4, 2)
            b = (box[0] + j[0], box[1] + j[1], box[2] + j[0], box[3] + j[1])
            crop = crop_resize(frame, b, size)
            crop = np.clip(crop.astype(np.float32) + rng.normal(0, 3, crop.shape), 0, 255).astype(np.uint8)
            feats.append(backbone(preprocess_batch(crop[None])))
        frac = max(box[2] - box[0], box[3] - box[1]) / float(min(frame.shape[:2]))
        thumb = cv2.resize(frame[box[1]:box[3], box[0]:box[2]], (64, 64), interpolation=cv2.INTER_AREA)
        data.add_shots(label, np.concatenate(feats, 0), frac, thumb)

    t0 = time.time()
    register((60, 60, 156, 156), "objA")
    register((300, 200, 396, 296), "objB")
    print(f"[1] 学习两类别完成: {time.time()-t0:.2f}s, labels={data.labels}, "
          f"shots={[len(s) for s in data.shots]}")

    # 手动框识别
    for box, want in (((70, 70, 150, 150), "objA"), ((310, 210, 390, 290), "objB")):
        crop = crop_resize(frame, box, size)
        probas = clf.predict_batch(backbone(preprocess_batch(crop[None])), data)[0]
        pred = data.labels[int(probas.argmax())]
        print(f"[2] 识别 {box} -> {pred} (p={probas.max():.3f})  期望 {want}")
        assert pred == want, f"识别错误: {pred} != {want}"
        assert probas.max() > 0.6, f"置信度过低: {probas.max()}"

    # EMA 平滑
    p = clf.predict_batch(backbone(preprocess_batch(crop_resize(frame, (70, 70, 150, 150), size)[None])), data)[0]
    for _ in range(5):
        p = clf.smooth(p, p)
    assert p.argmax() == data.labels.index("objA")

    # 自动扫描
    t0 = time.time()
    short_side = min(frame.shape[1], frame.shape[0])
    boxes, seen = [], set()
    for f in data.class_scan_fracs():
        for s in (1.0, 0.75):
            side = int(f * s * short_side)
            if side < 16 or side in seen:
                continue
            seen.add(side)
            boxes += build_scan_boxes(480, 360, side, 40)
    crops = np.stack([crop_resize(frame, b, size) for b in boxes])
    feat = backbone(preprocess_batch(crops))
    probas = clf.predict_batch(feat, data)
    dt = time.time() - t0
    i = int(probas.max(axis=1).argmax())
    bx = boxes[i]; pred = data.labels[int(probas[i].argmax())]
    print(f"[3] 扫描 {len(boxes)} 窗口 ({dt*1000:.0f}ms) -> 最佳窗口 {bx} -> {pred} "
          f"(p={probas[i].max():.3f})")
    truth = {"objA": (60, 60, 156, 156), "objB": (300, 200, 396, 296)}[pred]
    ix = min(bx[2], truth[2]) - max(bx[0], truth[0])
    iy = min(bx[3], truth[3]) - max(bx[1], truth[1])
    assert ix > 30 and iy > 30, f"扫描窗口未对准物体: {bx}"
    assert probas[i].max() > 0.6

    # Serialize in memory: self-test must never create or delete learning files.
    from io import BytesIO
    from unittest.mock import patch
    buffer = BytesIO()
    def capture_snapshot(_path, payload):
        np.savez(buffer, **payload)
        return True, '', 'memory://self-test'
    data.class_bgs[0] = np.ones(data.shots[0].shape[1], np.float32)
    with patch.dict(globals(), {'save_archive_snapshot': capture_snapshot}):
        ok, info = data.save("self-test-memory.npz")
    assert ok
    data2 = LearnedData()
    with np.load(BytesIO(buffer.getvalue()), allow_pickle=False) as payload:
        ok, info = data2._load_payload(payload)
    assert ok and data2.labels == data.labels
    for s1, s2 in zip(data.shots, data2.shots):
        assert np.allclose(s1, s2, atol=1e-4), "保存/加载特征不一致"
    assert len(data2.thumbs) == len(data.labels) and data2.thumbs[0].any(), "缩略图未持久化"
    assert data2.class_bg(0) is not None, "背景负样本未持久化"
    assert all(data2.class_bg(i) is None for i in range(1, len(data.labels)))
    print(f"[4] 内存序列化往返一致（含缩略图/背景负样本, {info}）")

    print("=== SELF-TEST OK ===")
    return 0


# ----------------------------------------------------------------------------
def print_banner():
    print("=" * 56)
    print("  PEFSL 框选式小样本自学习 + 识别演示")
    print("-" * 56)
    print("  A 模式: 框选物体 -> 输入标签 -> 自动采集特征完成学习")
    print("  B 模式: 自动定位并框出已学物体 / 框选区域识别 / s:定位开关")
    print("  +/- 键: 随时调整每次框选采集的帧数(右上角显示当前值)")
    print("  按键: a/b 切换模式 | s 自动扫描 | c 清除识别框")
    print("        r 清空学习数据 | w 保存 | l 加载 | q 退出")
    print("        n 标记负样本(框错了就按) | d 存这一帧的诊断证据(排查用)")
    print("  提示: 窗口需获得焦点; 输入标签请用英文输入法")
    print("=" * 56)


def get_args():
    p = argparse.ArgumentParser(
        description="PEFSL 框选式小样本自学习与识别演示 (A=学习 B=识别)",
        formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--confirmed-template-reacquire", action="store_true",
                   help="Experimental: recheck a confirmed learned template before reacquisition search")
    p.add_argument("--camera-id", type=int, default=0)
    p.add_argument("--camera-resolution", type=str, default="640x480")
    p.add_argument("--output-resolution", type=str, default="800x600")
    p.add_argument("--resolution-input", type=int, default=32,
                   help="骨干网络输入分辨率 (与权重匹配, 默认 32)")
    p.add_argument("--backbone", type=str, default="resnet9", choices=["resnet9", "resnet12"])
    p.add_argument("--path-pytorch-weight", type=str,
                   default=os.path.join(SCRIPT_DIR, "resnet9_strided_16fmaps.pt"))
    p.add_argument("--device-pytorch", type=str, default="cpu")
    p.add_argument("--cpu-threads", type=int, default=2,
                   help="CPU 推理/OpenCV 线程数；小图默认 2，减少线程调度造成的卡顿")
    p.add_argument("--sync-search", action="store_true",help="使用同步全局搜索作比较；默认异步搜索")
    p.add_argument("--search-hz", type=float, default=8.,help="异步搜索最高频率；光流与当前帧核验仍逐帧执行")
    p.add_argument("--classifier-type", type=str, default="ncm", choices=["ncm", "knn"],
                   help="PEFSL 分类器: ncm(最近类均值,默认) 或 knn")
    p.add_argument("--number-neiboors", type=int, default=5, help="knn 邻居数")
    p.add_argument("--shots-per-box", type=int, default=60,
                   help="每次框选采集帧数的启动默认值，运行中可随时按 +/- 调整 (步长5, 范围5~200)")
    p.add_argument("--min-box-size", type=int, default=20, help="框选最小边长(显示像素)")
    p.add_argument("--max-reg-frac", type=float, default=0.0,
                   help="Optional registration size limit relative to frame short side; 0 disables it. Keep the complete object.")
    p.add_argument("--ring-skin-th", type=float, default=0.25,
                   help="白/灰类新锁定的'周围没有手'阈值: 目标框外圈(外扩半个框)"
                        "肤色占比低于此值 -> 判'周围没有手'、不锁定。这个演示学的是"
                        "**手里举着**的物体, 真目标旁必有手; 而白衣服/白墙/白椅旁没有。"
                        "逐帧实测: 真涂改带外圈肤色 0.32~0.42, 背景误检 0.09~0.20 -> "
                        "0.25 分开两拨。旧值 0.05 太松等于没拦。设 0 关闭此闸门")
    p.add_argument("--closeup-ring-skip", type=float, default=0.50,
                   help="白/灰类就近举近时的**'周围没有手'豁免**: 候选框长边 >= 该比例×"
                        "画面短边时, 跳过'外圈必须有手'这道门。因为物体举到镜头前时框会"
                        "占满画面, 外圈是物体自身边缘而不是手, 该判据结构性失效(实测真"
                        "涂改带三关满分 87/87/87 却因 ring_skin=0.09 被拒 -> '只能认出一"
                        "瞬间')。默认 0.50(必须真的占满画面才豁免)。0.35 太松: 实测一张"
                        "248×100(0.41×短边)的'胸前白衬衫'误检框被误豁免, 而它的环带肤色"
                        "只有 0.09 —— 本来正好被这道门拦住。设 0 关闭豁免(恢复旧行为)")
    p.add_argument("--track-size-floor", type=float, default=0.35,
                   help="跟踪期**尺寸地板**: 锚定框长边不得低于 '该类注册长边 × 此比例'。"
                        "防止尺寸自适应把框一路缩小成细条(实测: 白类被缩到注册长边的 20%"
                        "=62×24, 变成衬衫上的一条细条)。默认 0.35(低于最小扫描档 0.45, 不"
                        "会与扫描网格打架); 设 0 关闭")
    p.add_argument("--white-win-skin-th", type=float, default=0.30,
                   help="白/灰类**新落锁**时的'窗口内肤色'上限: 框内肤色占比高于此值"
                        "-> 判为'框到了人身上的白块/背景人的衣服', 不锁定。实测误检"
                        "窗口肤色 0.25~0.55, 真白物体 0.02~0.07, 0.30 卡在空档里。"
                        "只作用于新落锁(跟踪期仍用 0.65), 不会造成'锁定后总是丢失目标'。"
                        "**若真物体被误拒(手挡住了物体), 设 0 关闭或调高(如 0.5)**")
    p.add_argument("--neg-abs-th", type=float, default=0.90,
                   help="白/灰类的**绝对负样本门**: 窗口与任一负样本的相似度 >= 此值就"
                        "掩蔽本类。相对规则(窗口'比本类更像负样本')对'白物体 vs 白衣服'"
                        "无效(两者一样像), 所以白/灰类需要这条绝对门: 负样本标的是**具体"
                        "误检位置**, 同处窗口相似度接近 1 会稳过; 真物体即使也白, 通常达不到。"
                        "默认 0.90。**若真涂改带也被误掩蔽(举着却认不出), 把它调高(如 0.95)**; "
                        "设 0 关闭。仅对'白/灰类'生效(注册直方图极低饱和占比>0.60)")
    p.add_argument("--auto-neg-hits", type=int, default=0,
                   help="自动负样本: 同一块地方被反复(达此次数)释放后, 自动把该区域"
                        "登记成负样本(仅白/灰类, 上限 12 个, 按 r 清空/按 w 持久化)。"
                        "默认 0 关闭，防止真目标因遮挡/漏检被错误登记；可显式开启实验")
    p.add_argument("--white-max-contrast", type=float, default=0.0,
                   help="白/灰类的**灰度对比度上限**(倍率): 窗口灰度对比度 > 注册值×此倍率"
                        "就拒绝。实测误检的白衣服块 = 注册值的 1.8 倍。默认 0=关闭; "
                        "想试可设 1.6~1.8(偏紧, 可能误伤对比度高的姿态)")
    p.add_argument("--max-scan-windows", type=int, default=60, help="全局扫描窗口数上限(按尺度均分; 前景预过滤后实际窗口数远少于上限)")
    p.add_argument("--scan-interval", type=int, default=4,
                   help="全局搜索节奏(每 N 帧; 跟踪不受此限制, 锁定后每帧跟踪)。"
                        "2→4: 全局搜索+精化网格是帧率大头, 降频后帧率明显提升")
    p.add_argument("--scan-scales", type=str, default="1.0,0.8,0.6,0.45,0.30,0.20,1.4,1.7,2.2,3.0",
                   help="自动扫描窗口尺度(相对注册框尺寸; 0.30~0.45 覆盖举得比注册时远的小物体, 1.4~3.0 覆盖举得近的大物体)")
    p.add_argument("--scan-cos-th", type=float, default=0.56,
                   help="候选筛选门槛基准(绝对余弦相似度0~1)。候选进入精化的"
                        "门槛 = 此值+0.02; 最终锁定门槛另有分级(小目标0.58/"
                        "一般0.62)。调低=更易检出, 调高=更不易误检")
    p.add_argument("--scan-marg-th", type=float, default=0.58,
                   help="颜色门第二道: 饱和度/亮度单通道分布的距离阈值(0=关闭)。"
                        "三维直方图在只有一个通道不同时会被稀释(格子衬衫白块 vs "
                        "白色物体), 单通道分布更敏感。0.45 会误伤真物体——书换角度"
                        "时封面内容(文字/黄色条)占比变了, 单通道分布跟着漂, 就被"
                        "整类掩蔽('书要换角度才认得出'); 0.58 只挡明显不符的")
    p.add_argument("--scan-hist-th", type=float, default=0.68,
                   help="颜色门控阈值(HSV直方图Bhattacharyya距离, 0=相同, 越小越严); "
                        "0.80 太松会放行颜色不符的背景/衣物, 0.5 太严会掩蔽浅色物体; "
                        "0.68 折中(0.62 会掩蔽掉真物体颜色略有偏移时的最佳窗口)")
    p.add_argument("--save-path", type=str,
                   default=os.path.join(SCRIPT_DIR, "bbox_demo_data_v2.npz"))
    p.add_argument("--diag-dir", type=str, default="",
                   help="按 d 存诊断证据的目录(渲染帧 png / 原图 png / diag.txt 追加)。"
                        "留空则默认 <脚本目录>/bbox_diag。这些是排查用的临时文件, 可随时删")
    p.add_argument("--lang", type=str, default="auto", choices=["auto", "zh", "en"])
    p.add_argument("--self-test", action="store_true", help="无摄像头自检并退出")
    args = p.parse_args()
    args.camera_resolution = parse_wh(args.camera_resolution)
    args.output_resolution = parse_wh(args.output_resolution)
    args.resolution_input = (args.resolution_input, args.resolution_input)
    args.scan_scales = tuple(float(s) for s in args.scan_scales.split(","))
    return args


def main():
    args = get_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    if args.self_test:
        return run_self_test(args)

    print_banner()
    renderer = TextRenderer(args.lang)
    if not os.path.exists(args.path_pytorch_weight):
        print(f"[error] 权重文件不存在: {args.path_pytorch_weight}")
        return 1
    backbone = load_backbone(args)
    data = LearnedData()
    if os.path.exists(args.save_path):
        print(f"[hint] 检测到已保存数据 {args.save_path}，启动后按 l 可加载")
    clf = PefslClassifier(args.classifier_type, args.number_neiboors)
    try:
        cap = init_camera(args)
    except RuntimeError as e:
        print(f"[error] {e}")
        return 1
    app = App(args, backbone, data, clf, renderer)
    app.run(cap)
    return 0


if __name__ == "__main__":
    sys.exit(main())
