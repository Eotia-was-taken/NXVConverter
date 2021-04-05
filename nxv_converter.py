<from typing import List
import sys
import numpy as np
import argparse
from moviepy.editor import *
import logging
from moviepy.audio.AudioClip import AudioArrayClip
import pydub
import io
from pydub.playback import play

"""
    Based on the format specified on: https://wiki.multimedia.cx/index.php/NXV
"""
"""
 bytes  0-11   ASCIIZ  magic ("NXV File")
 bytes 12-19   ASCIIZ  version ("1.0.0", "3.0.1" or "3.0.2")
 byte     20   width (pixels)
 byte     21   height (pixels)
 byte     22   always 0
 byte  23-511  unknown but required for playback (see below)
"""

"""
 uint curSequence = 0
 while !eof
   u8[500]        audio payload                -> 0:500
   u8[4]          unknown                      -> 500:504
   be32           sequence number              -> 504:508, BIG ENDIAN 32
   le16           length (bytes)               -> 508: 510, LITTLE ENDIAN 16
   le16           unknown (pixels == bytes/2?) -> 510:512 
   if (sequence == curSequence)              
     curSequence += 1
     u8[length]     video payload
"""

"""
    Video is encoded in 16 bit RGB565 
"""
class VideoFrame():
    def __init__(self, byte_array, height, width):
        # 16 bit RGB565
        self._byte_size = height * width * 2
        # The RGB565 format is converted to RGB888 here for easy convenience.
        self._payload = byte_array.view(">u2")
        # Normalise from 5-6 bit range to 8 bit range.
        R = ((self._payload[:height * width] >> 11) & 0x1F).astype(np.uint8) * 256 / (2 ** 5) # 5 Bits 
        G = ((self._payload[:height * width] >> 5)  & 0x3F).astype(np.uint8) * 256 / (2 ** 6) # 6 Bits
        B = ((self._payload[:height * width])       & 0x1F).astype(np.uint8) * 256 / (2 ** 5) # 5 Bits

        R = R.reshape((height, width))
        G = G.reshape((height, width))
        B = B.reshape((height, width))

        self._payload = np.zeros((height, width, 3), dtype=np.uint8)
        
        self._payload[:,:,2] = R
        self._payload[:,:,1] = G
        self._payload[:,:,0] = B

    @property
    def payload(self):
        return self._payload


class AudioFrame():
    def __init__(self, byte_array: np.array):
        # 500 bytes of audio payload
        self._payload = byte_array[:500]
        # unknown [500:504]
        self._sequence_number = int.from_bytes(byte_array[504: 508], "big", signed=False)
        self._video_length = int.from_bytes(byte_array[508: 510], "little", signed=False)
        # unknown [510:512]

    @property
    def payload(self):
        return self._payload
    
    @property
    def sequence_number(self):
        return self._sequence_number
    
    @property
    def video_length(self):
        return self._video_length


class NXVPackage():
    def __init__(self, file_name: str, output_file: str, verbose: bool):
        byte_array = read_file(file_name)
        self._header = Header(byte_array)
        self._video_frame_byte_size = self.header.width * self.header.height * 2
        self.broken_frames = 0
        self.output_file = output_file
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG if verbose else logging.WARNING)

        self.logger.debug("Header: version={}, width={}, height={}, video_frame_byte_size={}".format(self.header.version,
                                                                                                    self.header.width,
                                                                                                    self.header.height,
                                                                                                    self.video_frame_byte_size))
        self.read_frames(byte_array[512:])
        self.logger.debug("Video length: {} frames with {} FPS. The video is {} seconds long.".format(len(self._video_frames),
                                                                                                     self.header.frame_rate, 
                                                                                                     self.header.frame_rate * len(self._video_frames)))
        self.logger.debug("{} video frames were broken.".format(self.broken_frames))
        self.logger.debug("Saving file to {}".format(output_file))

        self.save_video_file()


    def convert_audio(self) -> AudioArrayClip:
        audio_sequence = list(map(lambda x: x.payload, self._audio_frames))
        audio_sequence = np.concatenate(audio_sequence)
        bytes_io = io.BytesIO(bytearray(audio_sequence))
        audio_clip = pydub.AudioSegment.from_file(bytes_io, format="mp3")
        audio_frame_rate = audio_clip.frame_rate
        audio_channels = audio_clip.channels
        audio_clip = np.asarray(audio_clip.get_array_of_samples(), dtype=np.int64)
        audio_clip = audio_clip / max(audio_clip)
        audio_clip = audio_clip.reshape((audio_clip.shape[0] // audio_channels, audio_channels))
        audio_clip = AudioArrayClip(audio_clip, fps=audio_frame_rate)

        return audio_clip

    def save_video_file(self):
        image_sequence = list(map(lambda x: x.payload, self._video_frames))
        video_clip = ImageSequenceClip(image_sequence, self.header.frame_rate)
        video_clip.audio = self.convert_audio()
        video_clip.write_videofile(self.output_file)

    def read_frames(self, byte_array: np.array) -> None:
        self._audio_frames = []
        self._video_frames = []
        idx = 0
        current_sequence_number = 0

        while True:
            if idx  > len(byte_array):
                break

            audio_frame = AudioFrame(byte_array[idx: idx + 512])
            self._audio_frames.append(audio_frame)
            current_sequence_number += 1
            if current_sequence_number % self.header._audio_per_video_ratio == 0:
                # length of video segment is part of the audio_frame definition
                # length of the segment is given in bytes (really don't know why..)
                number_of_video_frames  = audio_frame.video_length //  (self.video_frame_byte_size)
                # Length of frame is smaller than a single frame, something broke here.
                if number_of_video_frames == 0:
                    self.broken_frames += 1
                    number_of_video_frames = 1
                for i in range(number_of_video_frames):
                    video_frame = VideoFrame(byte_array[idx + 512 + i * self.video_frame_byte_size:], self.header.height, self.header.width)
                    self._video_frames.append(video_frame)
                    idx += self.video_frame_byte_size

            idx += 512
         
    @property
    def header(self):
        return self._header

    @property
    def video_frame_byte_size(self):
        return self._video_frame_byte_size
    
    
class Header():
    def __init__(self, byte_array: np.array):
        self._version = Header._convert_version_to_string(byte_array[12:20])
        self._width = int(byte_array[20]) 
        self._height = int(byte_array[21])
        self._audio_per_video_ratio = byte_array[byte_array[0x17]]
        self._frame_rate = 32 // self.audio_per_video_ratio

    @staticmethod
    def _convert_version_to_string(byte_array: np.array) -> str:
        v = ""
        for byte in byte_array:
            if byte != 0:
                v += chr(byte)
        return v

    @property
    def frame_rate(self):
        return self._frame_rate
    
    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height
    
    @property
    def version(self):
        return self._version

    @property
    def audio_per_video_ratio(self):
        return self._audio_per_video_ratio
    

def read_file(file_name: str) -> np.array:
    with open(file_name, 'rb') as f:
        byte_array = f.read()
    
    return np.frombuffer(byte_array, dtype=np.uint8)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input NXV file.")
    parser.add_argument("-o", "--output", help="Location of the output video.", default="output.mp4")
    parser.add_argument("-v", "--verbose", help="Set verbose output option.", action="store_true")

    logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=0)
    args = parser.parse_args()
    file_name = args.input
    output_file = args.output
    verbose = args.verbose

    package = NXVPackage(file_name, output_file, verbose)

