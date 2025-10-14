"""FFMPEG command line converter"""

__Contributors__ = "Mike Goebel"
__version__ = "0.0"
__copyright__ = "Center for BioImage Informatics, University California, Santa Barbara"

import logging
from lxml import etree

from bq.util.locks import Locks
from bq.image_service.controllers.converter_base import ConverterBase, Format
from bq.util.compat import OrderedDict
from bq.image_service.controllers.exceptions import (
    ImageServiceException,
    ImageServiceFuture,
)
import subprocess
import json
import inspect
import shutil
import os

log = logging.getLogger("bq.image_service.converter_ffmpeg")


supported_formats = [
    ("MP4", "MPEG4", ["mp4"]),
    ("AVI", "Microsoft AVI", ["avi"]),
    (
        "WEBM",
        "WEBM",
        [
            "webm",
        ],
    ),
    ("MOV", "QuickTime Movie", ["mov"]),
]


def compute_new_size(imw, imh, w, h, keep_aspect_ratio, no_upsample):
    if no_upsample is True and imw <= w and imh <= h:
        return (imw, imh)

    if keep_aspect_ratio is True:
        if imw / float(w) >= imh / float(h):
            h = 0
        else:
            w = 0

    # it's allowed to specify only one of the sizes, the other one will be computed
    if w == 0:
        w = int(round(imw / (imh / float(h))))
    if h == 0:
        h = int(round(imh / (imw / float(w))))

    return (w, h)


class ConverterFfmpeg(ConverterBase):
    # All of this metadata on the converter is made up, should be fixed at some point
    # if this code becomes a significant part of bisqueConverterImgcnv
    installed = True
    version = [1, 2, 3]
    installed_formats = None
    name = "ffmpeg"
    required_version = "0.0.0"

    @classmethod
    def get_version(cls):
        return {
            "full": ".".join([str(i) for i in cls.version]),
            "numeric": cls.version,
            "major": cls.version[0],
            "minor": cls.version[1],
            "build": cls.version[2],
        }

    @classmethod
    def get_formats(cls):
        try:
            cls.installed_formats = OrderedDict()
            for name, fullname, exts in supported_formats:
                cls.installed_formats[name.lower()] = Format(
                    name=name,
                    fullname=fullname,
                    ext=exts,
                    reading=True,
                    writing=True,
                    multipage=True,
                    metadata=True,
                    samples=(0, 0),
                    bits=(8, 8),
                )
        except Exception as e:
            log.info("Get formats failed with error " + str(e))
        return cls.installed_formats

    @classmethod
    def get_installed(cls):
        return True

    @classmethod
    def supported(cls, token, **kw):
        """return True if the input file format is supported"""
        ifnm = token.first_input_file()
        all_exts = set()
        for fmt in supported_formats:
            all_exts.add(*fmt[2])
        return ifnm.split(".")[-1].lower() in all_exts

    @classmethod
    def convert(cls, token, ofnm, fmt=None, extra=None, **kw):
        # log.info(f"--- extra parameters: {extra} kw: {kw}")
        ifnm = token.first_input_file()
        
        # Check if this is a multi-frame DICOM conversion (z-stack to video)
        is_multiframe_dicom = (
            token.dims.get("image_num_z", 1) > 1 and 
            token.dims.get("storage", "") == "multi_file_series" and
            fmt.lower() in ["webm", "mp4", "avi", "mov"]
        )
        
        log.info(f"FFmpeg convert: multiframe_dicom={is_multiframe_dicom}, z={token.dims.get('image_num_z')}, storage={token.dims.get('storage')}, fmt={fmt}")
        
        with Locks(ifnm, ofnm, failonexist=True) as l:
            if l.locked:
                # log.info('\n\n\n\nConverting video1:\n\n\n\n')
                ifnm = token.first_input_file()
                imw = token.dims["image_num_x"]
                imh = token.dims["image_num_y"]
                ##log.info('\n\n\n\nConverting video2:\n\n\n\n')
                ind_rs = [i for i, v in enumerate(extra) if v == "-resize"]
                resize = True
                if len(ind_rs) != 1:
                    resize = False
                else:
                    # log.info('\n\n\n\nConverting video2.6:\n\n\n\n')
                    rs_string = extra[ind_rs[0] + 1]
                    width, height = [int(i) for i in rs_string.split(",")[:2]]

                    # ind_rs = ind_rs[0] + 1

                # log.info('\n\n\n\nConverting video3:\n\n\n\n')

                if is_multiframe_dicom:
                    # For multi-frame DICOM sequences, always extract frames fresh in temporary directory
                    log.info("Creating video from DICOM file by extracting frames fresh")
                    return cls._convert_dicom_via_frames(token, ofnm, fmt, extra, **kw)
                elif ifnm.lower().endswith('.dcm'):
                    # For single DICOM files, try standard FFmpeg conversion first
                    log.info("Using standard FFmpeg conversion for single DICOM file")
                    
                    if resize:
                        w_out, h_out = compute_new_size(
                            imw, imh, width, height,
                            keep_aspect_ratio=True, no_upsample=True,
                        )
                        cmd = [
                            "ffmpeg", "-y", "-hide_banner", "-threads", "8", "-loglevel", "error",
                            "-i", ifnm,
                            "-vf", f"scale={w_out}:{h_out}",
                            ofnm,
                        ]
                    else:
                        cmd = [
                            "ffmpeg", "-y", "-hide_banner", "-threads", "8", "-loglevel", "error",
                            "-i", ifnm,
                            ofnm,
                        ]
                    
                    log.info(f"-----Executing DICOM command: {' '.join(cmd)}")
                    try:
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                        if result.returncode == 0 and os.path.exists(ofnm):
                            log.info(f"DICOM video conversion successful: {ofnm}")
                            return ofnm
                        else:
                            log.warning(f"DICOM FFmpeg conversion failed: {result.stderr}")
                            # Fall back to frame extraction approach
                            return cls._convert_dicom_via_frames(token, ofnm, fmt, extra, **kw)
                    except subprocess.TimeoutExpired:
                        log.error("DICOM FFmpeg conversion timed out")
                        # Fall back to frame extraction approach
                        return cls._convert_dicom_via_frames(token, ofnm, fmt, extra, **kw)
                    except Exception as e:
                        log.error(f"DICOM FFmpeg conversion failed with exception: {e}")
                        # Fall back to frame extraction approach
                        return cls._convert_dicom_via_frames(token, ofnm, fmt, extra, **kw)
                else:
                    # Standard single-file video conversion
                    if resize:
                        w_out, h_out = compute_new_size(
                            imw,
                            imh,
                            width,
                            height,
                            keep_aspect_ratio=True,
                            no_upsample=True,
                        )
                        cmd = [
                            "ffmpeg",
                            "-y",
                            "-hide_banner",
                            "-threads",
                            "8",
                            "-loglevel",
                            "error",
                            "-i",
                            ifnm,
                            "-vf",
                            "scale=" + str(w_out) + ":" + str(h_out),
                            ofnm,
                        ]
                    else:
                        cmd = [
                            "ffmpeg",
                            "-y",
                            "-hide_banner",
                            "-threads",
                            "8",
                            "-loglevel",
                            "error",
                            "-i",
                            ifnm,
                            ofnm,
                        ]

                single_cmd = " ".join(cmd)
                log.info(f"-----Executing command: {single_cmd}")

                try:
                    process = subprocess.run(
                        single_cmd,
                        shell=True,
                        stdin=subprocess.DEVNULL,  # avoid blocking stdin
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        env=os.environ.copy(),  # inherit full shell env
                        text=True,
                    )

                    # Cleanup temporary files if they were created
                    if is_multiframe_dicom:
                        concat_file = ofnm + ".frames.txt"
                        if os.path.exists(concat_file):
                            try:
                                os.remove(concat_file)
                            except:
                                pass
                        # Also clean up any temporary frame directories that might exist
                        import glob
                        temp_dirs = glob.glob("/tmp/video_frames_*")
                        for temp_dir in temp_dirs:
                            try:
                                shutil.rmtree(temp_dir)
                            except:
                                pass

                    if process.returncode == 0 and os.path.exists(ofnm):
                        log.info(f"Video conversion successful using {ofnm}")
                        return ofnm
                    else:
                        log.warning(
                            f"Video conversion failed with {ofnm}: stderr={process.stderr}, stdout={process.stdout}, returncode={process.returncode}"
                        )
                        # Clean up partial file if it exists
                        if os.path.exists(ofnm):
                            try:
                                os.remove(ofnm)
                            except:
                                pass

                except Exception as e:
                    log.error(f"FFmpeg execution failed: {e}")
                    # Clean up temporary files if they were created
                    if is_multiframe_dicom:
                        concat_file = ofnm + ".frames.txt"
                        if os.path.exists(concat_file):
                            try:
                                os.remove(concat_file)
                            except:
                                pass
                        # Also clean up any temporary frame directories that might exist
                        import glob
                        temp_dirs = glob.glob("/tmp/video_frames_*")
                        for temp_dir in temp_dirs:
                            try:
                                shutil.rmtree(temp_dir)
                            except:
                                pass

                return None

            elif l.locked is False:
                raise ImageServiceFuture((1, 15))

    @classmethod
    def _convert_dicom_via_frames(cls, token, ofnm, fmt=None, extra=None, **kw):
        """
        Fallback method for DICOM files when FFmpeg direct conversion fails.
        Uses imgcnv to extract frames and then creates video from those frames.
        """
        log.info("Attempting DICOM conversion via frame extraction")
        
        ifnm = token.first_input_file()
        imw = token.dims["image_num_x"]
        imh = token.dims["image_num_y"]
        
        # Determine resize parameters
        ind_rs = [i for i, v in enumerate(extra) if v == "-resize"]
        resize = len(ind_rs) == 1
        if resize:
            rs_string = extra[ind_rs[0] + 1]
            width, height = [int(i) for i in rs_string.split(",")[:2]]
        
        try:
            # Import imgcnv converter for frame extraction
            from .converter_imgcnv import ConverterImgcnv
            
            # Check if this is a multi-file series
            is_multifile_series = token.dims.get("storage", "") == "multi_file_series" and isinstance(token.input, list)
            
            if is_multifile_series:
                # For multi-file series, each file is a frame
                files_to_process = token.input
                num_frames = len(files_to_process)
                log.info(f"Multi-file DICOM series with {num_frames} files")
            else:
                # For single multi-frame DICOM, calculate frames from dimensions
                num_frames = token.dims.get("image_num_z", 1) * token.dims.get("image_num_t", 1)
                files_to_process = [ifnm]  # Single file
                log.info(f"Single DICOM file with {num_frames} frames (z={token.dims.get('image_num_z', 1)}, t={token.dims.get('image_num_t', 1)})")
            
            if num_frames <= 1:
                log.warning("DICOM has only 1 frame, cannot create video")
                return None
            
            log.info(f"DICOM dimensions: {token.dims}")
            
            # Create temporary directory for extracted frames
            import tempfile
            
            temp_dir = tempfile.mkdtemp(prefix="dicom_frames_")
            log.info(f"Created temporary directory: {temp_dir}")
            
            extracted_frames = []
            
            try:
                if is_multifile_series:
                    # Extract one frame from each file in the series
                    for file_idx, dicom_file in enumerate(files_to_process):
                        frame_filename = f"frame_{file_idx:04d}.png"
                        frame_path = os.path.join(temp_dir, frame_filename)
                        
                        try:
                            # Build imgcnv command to extract frame 1 from this file
                            command = [
                                'imgcnv',
                                '-i', dicom_file,
                                '-o', frame_path,
                                '-page', '1',  # Always extract first frame from each file
                                '-t', 'png',
                                '-enhancemeta'  # Enhance metadata for proper DICOM display
                            ]
                            
                            # Add depth conversion if needed for proper brightness/contrast
                            if token.dims.get('image_pixel_depth', 16) != 8:
                                command.extend(['-depth', '8,d,u'])  # Convert to 8-bit with dynamic range adjustment
                            
                            # Use the proper imgcnv interface
                            from .converter_imgcnv import call_imgcnvlib
                            retcode, out = call_imgcnvlib(command)
                            
                            if retcode == 0 and os.path.exists(frame_path):
                                extracted_frames.append(frame_path)
                                log.debug(f"Extracted frame {file_idx} from {dicom_file}: {frame_path}")
                            else:
                                log.warning(f"Failed to extract frame from {dicom_file}: return code {retcode}, output: {out}")
                        except Exception as e:
                            log.warning(f"Error extracting frame from {dicom_file}: {e}")
                            continue
                else:
                    # Extract each frame from the single multi-frame DICOM file
                    for frame_idx in range(num_frames):
                        frame_filename = f"frame_{frame_idx:04d}.png"
                        frame_path = os.path.join(temp_dir, frame_filename)
                        
                        try:
                            # Build imgcnv command to extract specific frame
                            command = [
                                'imgcnv',
                                '-i', ifnm,
                                '-o', frame_path,
                                '-page', str(frame_idx + 1),  # imgcnv uses 1-based indexing
                                '-t', 'png',
                                '-enhancemeta'  # Enhance metadata for proper DICOM display
                            ]
                            
                            # Add depth conversion if needed for proper brightness/contrast
                            if token.dims.get('image_pixel_depth', 16) != 8:
                                command.extend(['-depth', '8,d,u'])  # Convert to 8-bit with dynamic range adjustment
                            
                            # Use the proper imgcnv interface
                            from .converter_imgcnv import call_imgcnvlib
                            retcode, out = call_imgcnvlib(command)
                            
                            if retcode == 0 and os.path.exists(frame_path):
                                extracted_frames.append(frame_path)
                                log.debug(f"Extracted frame {frame_idx}: {frame_path}")
                            else:
                                log.warning(f"Failed to extract frame {frame_idx}: return code {retcode}, output: {out}")
                        except Exception as e:
                            log.warning(f"Error extracting frame {frame_idx}: {e}")
                            continue
                
                if len(extracted_frames) < 2:
                    log.error(f"Only extracted {len(extracted_frames)} frames, need at least 2 for video")
                    return None
                
                log.info(f"Successfully extracted {len(extracted_frames)} frames")
                
                # Create FFmpeg concat file
                concat_file = os.path.join(temp_dir, "frames.txt")
                with open(concat_file, 'w') as f:
                    for frame_path in extracted_frames:
                        abs_frame_path = os.path.abspath(frame_path)
                        f.write(f"file '{abs_frame_path}'\n")
                        f.write("duration 0.1\n")  # 10 FPS
                    # Repeat last frame briefly
                    if extracted_frames:
                        abs_last_path = os.path.abspath(extracted_frames[-1])
                        f.write(f"file '{abs_last_path}'\n")
                
                # Build FFmpeg command
                if resize:
                    w_out, h_out = compute_new_size(
                        imw, imh, width, height,
                        keep_aspect_ratio=True, no_upsample=True,
                    )
                    cmd = [
                        "ffmpeg", "-y", "-hide_banner", "-threads", "8", "-loglevel", "error",
                        "-f", "concat", "-safe", "0", "-i", concat_file,
                        "-vf", f"scale={w_out}:{h_out}",
                        "-pix_fmt", "yuv420p", "-c:v", "libvpx-vp9", "-crf", "30",
                        ofnm,
                    ]
                else:
                    cmd = [
                        "ffmpeg", "-y", "-hide_banner", "-threads", "8", "-loglevel", "error",
                        "-f", "concat", "-safe", "0", "-i", concat_file,
                        "-pix_fmt", "yuv420p", "-c:v", "libvpx-vp9", "-crf", "30",
                        ofnm,
                    ]
                
                log.info(f"Executing frame-based DICOM conversion: {' '.join(cmd)}")
                
                # Execute FFmpeg command
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0 and os.path.exists(ofnm):
                    log.info(f"DICOM frame-based conversion successful: {ofnm}")
                    return ofnm
                else:
                    log.error(f"Frame-based DICOM conversion failed: {result.stderr}")
                    return None
                    
            finally:
                # Clean up temporary files
                try:
                    shutil.rmtree(temp_dir)
                    log.debug(f"Cleaned up temporary directory: {temp_dir}")
                except Exception as e:
                    log.warning(f"Failed to clean up temporary directory {temp_dir}: {e}")
                    
        except ImportError:
            log.error("imgcnv converter not available for frame extraction")
            return None
        except Exception as e:
            log.error(f"DICOM frame extraction conversion failed: {e}")
            return None

    @classmethod
    def thumbnail(cls, token, ofnm, width, height, **kw):
        ifnm = token.first_input_file()
        with Locks(ifnm, ofnm, failonexist=True) as l:
            if l.locked is False:
                raise ImageServiceFuture((1, 15))
            # log.info('Creating thumbnail:')

            ifnm = token.first_input_file()
            imw = token.dims["image_num_x"]
            imh = token.dims["image_num_y"]

            w_out, h_out = compute_new_size(
                imw, imh, width, height, keep_aspect_ratio=True, no_upsample=True
            )

            cmd = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-threads",
                "1",
                "-loglevel",
                "error",
                "-i",
                ifnm,
                "-vframes",
                "1",
                "-s",
                str(w_out) + "x" + str(h_out),
                ofnm,
            ]

            process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
            output, error = process.communicate()

            if error is not None:
                return None

            return ofnm

    @classmethod
    def slice(cls, token, ofnm, z, t, roi=None, **kw):
        ifnm = token.first_input_file()
        with Locks(ifnm, ofnm, failonexist=True) as l:
            if l.locked is False:
                raise ImageServiceFuture((1, 15))

            # # log.info('Creating slice:')
            # cmd = [
            #     "ffmpeg",
            #     "-y",
            #     "-hide_banner",
            #     "-threads",
            #     "1",
            #     "-loglevel",
            #     "error",
            #     "-i",
            #     ifnm,
            #     "-vf",
            #     "select=eq(n\\," + str(t[0] - 1) + ")",
            #     "-vframes",
            #     "1",
            #     "-compression_algo",
            #     "raw",
            #     "-pix_fmt",
            #     "rgb24",
            #     ofnm,
            # ]

            # Use time-based seek instead of NAL-fragile frame selection
            time_in_sec = (
                float(t[0]) / 25
            )  # assuming 25 fps, will need to adjust based on actual frame rate later
            cmd = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{time_in_sec:.3f}",
                "-i",
                ifnm,
                "-frames:v",
                "1",
                "-pix_fmt",
                "rgb24",
                ofnm,
            ]

            single_cmd = " ".join(cmd)
            log.info(f"-----Executing command: {single_cmd}")

            process = subprocess.run(
                single_cmd,
                shell=True,
                stdin=subprocess.DEVNULL,  # avoid blocking stdin
                env=os.environ.copy(),  # inherit full shell env
                text=True,
            )

            if process.returncode == 0 and os.path.exists(ofnm):
                log.info(f"Video conversion successful using of {ofnm}")
                return ofnm
            else:
                log.warning(f"Video conversion failed of {ofnm}: {process.stderr}")
                # Clean up partial file if it exists
                if os.path.exists(ofnm):
                    try:
                        os.remove(ofnm)
                    except:
                        pass

            return None

    @classmethod
    def tile(cls, token, ofnm, level, x, y, sz, **kw):
        return None

    @classmethod
    def info(cls, token, **kw):

        ifnm = token.first_input_file()
        if not cls.supported(token):
            return None

        with Locks(ifnm, failonread=(True)) as l:
            if l.locked is False:
                raise ImageServiceFuture((1, 15))

            cmd = [
                "ffprobe",
                "-hide_banner",
                "-threads",
                "4",
                "-loglevel",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                "-i",
                ifnm,
            ]

            process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
            output, error = process.communicate()

            if error is not None:
                return dict()

            data = json.loads(output)

            if "streams" not in list(data.keys()):
                return dict()

            if "format" not in list(data.keys()):
                return dict()

            vid_stream = [s for s in data["streams"] if s["codec_type"] == "video"][0]
            f_format = data["format"]

            try:
                cmd = ["file", ifnm]
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
                output, error = process.communicate()
                endian = output.decode().split("(")[1].split("-")[0]

            except Exception as e:
                endian = "little"

            out_dict = dict()
            out_dict["image_num_x"] = vid_stream["width"]
            out_dict["image_num_y"] = vid_stream["height"]
            out_dict["image_num_z"] = 1
            out_dict["converter"] = "ffmpeg"
            out_dict["format"] = f_format["format_name"]
            out_dict["image_num_resolution_levels"] = 0
            out_dict["raw_endian"] = endian
            out_dict["image_pixel_depth"] = 8
            out_dict["image_pixel_format"] = "unsigned integer"
            out_dict["image_mode"] = "RGB"
            out_dict["image_series_index"] = 0
            out_dict["image_num_p"] = 1
            out_dict["image_num_c"] = 3
            out_dict["image_num_series"] = 0
            out_dict["filesize"] = f_format["size"]
            # log.info("FOOBAR " + str(f_format) + '\n\n' + str(vid_stream))
            if "nb_frames" in list(vid_stream.keys()):
                out_dict["image_num_t"] = vid_stream["nb_frames"]
            else:
                duration = float(f_format["duration"])
                frame_rate = float(
                    float(vid_stream["avg_frame_rate"].split("/")[0])
                    / float(vid_stream["avg_frame_rate"].split("/")[1])
                )
                out_dict["image_num_t"] = int(duration * frame_rate)

            log.info(f"-----Video info: {out_dict}")

            return out_dict

    def meta(cls, token, **kw):
        return cls.info(token, **kw)


try:
    ConverterFfmpeg.init()
except Exception:
    log.warning("FFMPEG not available")
