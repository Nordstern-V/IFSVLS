#   Copyright (c) 2018, Xilinx, Inc.
#   All rights reserved.
#
#   Redistribution and use in source and binary forms, with or without
#   modification, are permitted provided that the following conditions are met:
#
#   1.  Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#   2.  Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#   3.  Neither the name of the copyright holder nor the names of its
#       contributors may be used to endorse or promote products derived from
#       this software without specific prior written permission.
#
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
#   THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
#   PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
#   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
#   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
#   OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
#   WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
#   OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
#   ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

__author__ = "Peter Ogden"
__copyright__ = "Copyright 2018, Xilinx"
__email__ = "2568365021@qq.com"

import contextlib
from pynq import DefaultHierarchy
from .frontend import VideoInFrontend, VideoOutFrontend
from .vdma import AxiVDMA
from .common import *


class VideoOut(DefaultHierarchy):
    """Wrapper for the output video pipeline.

    This wrapper assumes the following pipeline structure and naming

    axi_vdma -> frontend
    with vtc_out and axi_dynclk helper IP

    Attributes
    ----------
    frontend : pynq.lib.video.HDMIOutFrontend
        The HDMI frontend for mode setting

    """

    @staticmethod
    def checkhierarchy(description):
        if 'frontend' in description['hierarchies']:
            frontend_dict = description['hierarchies']['frontend']
        elif 'frontend' in description['ip']:
            frontend_dict = description['ip']['frontend']
        else:
            return False
        return issubclass(frontend_dict['driver'], VideoOutFrontend)

    def __init__(self, description, vdma=None):
        """Initialise the drivers for the pipeline

        Parameters
        ----------
        path : str
            name of the hierarchy containing all of the video blocks

        """
        super().__init__(description)
        self._vdma = vdma
        self._hdmi = self.frontend
        self.vd_mode= None

    def lcd_id_rd(self):
        """获取LCD屏ID

        //R7(M0);G7(M1);B7(M2);
        //M2:M1:M0
        //0 :0 :0	//4.3寸480*272 RGB屏,ID=0X4342
        //0 :0 :1	//7寸800*480 RGB屏,ID=0X7084
        //0 :1 :0	//7寸1024*600 RGB屏,ID=0X7016
        //1 :0 :0	//4.3寸800*480 RGB屏,ID=0X4384
        //1 :0 :1	//10.1寸1280*800 RGB屏,ID=0X1018
        //返回值:LCD ID:0,非法;其他值,ID;

        """
        lcd_id_f = self._hdmi._lcd_id
        idx = lcd_id_f.channel1.read()

        if idx == 1:
            lcd_id= 0x7084
            mode =  VideoMode(800, 480, 24)
        elif idx == 2:
            lcd_id= 0x7016
            mode =  VideoMode(1024, 600, 24)
        elif idx == 4:
            lcd_id= 0x4384
            mode =  VideoMode(800, 480, 24)
        elif idx == 5:
            lcd_id= 0x1018  #10.1寸屏,1280*800分辨率
            mode =  VideoMode(1280, 800, 24)
        else:
            lcd_id= 0x4342  #4.3寸屏,480*272分辨率
            mode =  VideoMode(480, 272, 24)

        self.vd_mode = mode
        return lcd_id

    def configure(self, is_lcd, mode=None):
        """Configure the pipeline to use the specified pixel format and size.

        If the pipeline is running it is stopped prior to the configuration
        being changed

        Parameters
        ----------
        is_lcd : is LCD ?
            1, is lcd, no need to input mode parameter
            0, is hdmi, need mode parameter
        mode : VideoMode
            The video mode to output

        """
        if self._vdma.writechannel.running:
            self._vdma.writechannel.stop()
        if is_lcd:
            mode = self.vd_mode
        self._hdmi.mode = mode
        self._vdma.writechannel.mode = mode
        self._hdmi.start()
        return self._closecontextmanager()

    def start(self):
        """Start the pipeline

        """
        self._vdma.writechannel.start()
        return self._stopcontextmanager()

    def stop(self):
        """Stop the pipeline

        """
        self._vdma.writechannel.stop()

    def close(self):
        """Close the pipeline an unintialise the drivers

        """
        self.stop()
        self._hdmi.stop()

    @contextlib.contextmanager
    def _stopcontextmanager(self):
        """Context Manager to stop the VDMA at the end of the block

        """
        yield
        self.stop()

    @contextlib.contextmanager
    def _closecontextmanager(self):
        """Context Manager to close the HDMI port at the end of the block

        """
        yield
        self.close()

    @property
    def colorspace(self):
        """Set the colorspace for the pipeline - can be done without
        stopping the pipeline

        """
        return self._color.colorspace

    @colorspace.setter
    def colorspace(self, new_colorspace):
        self._color.colorspace = new_colorspace

    @property
    def mode(self):
        """The currently configured video mode

        """
        return self._vdma.writechannel.mode

    @property
    def cacheable_frames(self):
        """Whether frames should be cacheable or non-cacheable

        Only valid if a VDMA has been specified

        """
        if self._vdma:
            return self._vdma.writechannel.cacheable_frames
        else:
            raise RuntimeError("No VDMA specified")

    @cacheable_frames.setter
    def cacheable_frames(self, value):
        if self._vdma:
            self._vdma.writechannel.cacheable_frames = value
        else:
            raise RuntimeError("No VDMA specified")

    def newframe(self):
        """Return an unintialised video frame of the correct type for the
        pipeline

        """
        return self._vdma.writechannel.newframe()

    def writeframe(self, frame):
        """Write the frame to the video output

        See AxiVDMA.MM2SChannel.writeframe for more details

        """
        self._vdma.writechannel.writeframe(frame)

    async def writeframe_async(self, frame):
        """Write the frame to the video output

        See AxiVDMA.MM2SChannel.writeframe for more details

        """
        await self._vdma.writechannel.writeframe_async(frame)


class VideoWrapper(DefaultHierarchy):
    """Hierarchy driver for the entire video subsystem.

    Exposes the input, output and video DMA as attributes. For most
    use cases the wrappers for the input and output pipelines are
    sufficient and the VDMA will not need to be used directly.

    Attributes
    ----------
    video_in : pynq.lib.video.VideoIn
        The Video input pipeline
    video_out : pynq.lib.video.VideoOut
        The Video output pipeline
    axi_vdma : pynq.lib.video.AxiVDMA
        The video DMA.

    """
    @staticmethod
    def checkhierarchy(description):
        in_pipeline = None
        out_pipeline = None
        dma = None
        for hier, details in description['hierarchies'].items():
            if details['driver'] == VideoOut:
                out_pipeline = hier

        for ip, details in description['ip'].items():
            if details['driver'] == AxiVDMA:
                dma = ip

        return (out_pipeline is not None and
                dma is not None)

    class VideoIn:
        """Wrapper for the input video pipeline.

        camera -> axi_vdma

        """

        def __init__(self, vdma):
            """Initialise the drivers for the pipeline

            Parameters
            ----------
            path : str
                name of the hierarchy containing all of the video blocks

            """
            self._vdma = vdma

        def configure(self, mode=None):
            """Configure the pipeline to use the specified mode.

            If the pipeline is running it is stopped prior to the configuration
            being changed

            Parameters
            ----------
            mode : VideoMode
                The video mode to input

            """
            if self._vdma.readchannel.running:
                self._vdma.readchannel.stop()
            if mode == None:
                input_mode = VideoMode(800, 480, 24)
            else:
                input_mode = mode
            self._vdma.readchannel.mode = input_mode
            return self._closecontextmanager()

        def start(self):
            """Start the pipeline

            """
            self._vdma.readchannel.start()
            return self._stopcontextmanager()

        def stop(self):
            """Stop the pipeline

            """
            self._vdma.readchannel.stop()

        @contextlib.contextmanager
        def _stopcontextmanager(self):
            """Context Manager to stop the VDMA at the end of the block

            """
            yield
            self.stop()

        @contextlib.contextmanager
        def _closecontextmanager(self):
            """Context Manager to close the HDMI port at the end of the block

            """
            yield
            self.close()

        def close(self):
            """Uninitialise the drivers, stopping the pipeline beforehand

            """
            self.stop()

        @property
        def colorspace(self):
            """The colorspace of the pipeline, can be changed without stopping
            the pipeline

            """
            return self._color.colorspace

        @colorspace.setter
        def colorspace(self, new_colorspace):
            self._color.colorspace = new_colorspace

        @property
        def mode(self):
            """Video mode of the input

            """
            return self._vdma.readchannel.mode

        @property
        def cacheable_frames(self):
            """Whether frames should be cacheable or non-cacheable

            Only valid if a VDMA has been specified

            """
            if self._vdma:
                return self._vdma.readchannel.cacheable_frames
            else:
                raise RuntimeError("No VDMA specified")

        @cacheable_frames.setter
        def cacheable_frames(self, value):
            if self._vdma:
                self._vdma.readchannel.cacheable_frames = value
            else:
                raise RuntimeError("No VDMA specified")

        def readframe(self):
            """Read a video frame

            See AxiVDMA.S2MMChannel.readframe for details

            """
            return self._vdma.readchannel.readframe()

        async def readframe_async(self):
            """Read a video frame

            See AxiVDMA.S2MMChannel.readframe for details

            """
            return await self._vdma.readchannel.readframe_async()

        def tie(self, output):
            """Mirror the video input on to an output channel

            Parameters
            ----------
            output : HDMIOut
                The output to mirror on to

            """
            self._vdma.readchannel.tie(output._vdma.writechannel)

    def __init__(self, description):
        super().__init__(description)
        out_pipeline = None
        dma = None
        for hier, details in description['hierarchies'].items():
            if details['driver'] == VideoOut:
                out_pipeline = hier
        for ip, details in description['ip'].items():
            if details['driver'] == AxiVDMA:
                dma = ip
        self.video_in = VideoWrapper.VideoIn(getattr(self, dma))
        getattr(self, out_pipeline)._vdma = getattr(self, dma)
