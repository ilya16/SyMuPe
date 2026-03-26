from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import psutil

from symupe.utils import ExplicitEnum
from .aligner import Aligner
from ..alignment import Alignment

PROGRAMS = [
    "midi2pianoroll",
    "SprToFmt3x",
    "Fmt3xToHmm",
    "ScorePerfmMatcher",
    "ErrorDetection",
    "RealignmentMOHMM",
    "MatchToCorresp"
]

FILE_DIRECTORIES = [
    "spr",
    "fmt3x",
    "hmm",
    "match",
    "alignments"
]


class DataType(ExplicitEnum):
    SCORE = "score"
    PERFORMANCE = "perf"
    ALIGNMENT = "align"


class FileType(ExplicitEnum):
    MIDI = "MIDI"
    SPR = "spr"
    FMT3X = "fmt3x"
    HMM = "hmm"
    PRE_MATCH = "pre-match"
    ERR_MATCH = "err-match"
    MATCH = "match"
    ERR_CORRESP = "err-corresp"
    CORRESP = "corresp"
    ALIGN = "align"


FILE_EXT = {
    FileType.MIDI: ".mid",
    FileType.SPR: "_spr.txt",
    FileType.FMT3X: "_fmt3x.txt",
    FileType.HMM: "_hmm.txt",
    FileType.PRE_MATCH: "_pre-match.txt",
    FileType.ERR_MATCH: "_err-match.txt",
    FileType.MATCH: "_match.txt",
    FileType.ERR_CORRESP: "_err-corresp.txt",
    FileType.CORRESP: "_corresp.txt",
    FileType.ALIGN: "_align.txt"
}


def run_subprocess_with_limits(
        command: list,
        timeout: float | None = 1000,
        memory_limit: int | None = 8_000_000_000
):
    start_time = time.perf_counter()

    process = subprocess.Popen(command)  # start the subprocess
    p = psutil.Process(process.pid)  # get the process object for the subprocess

    while True:
        # check if the process has completed
        if process.poll() is not None:
            return 0

        # check timeout
        if timeout is not None and time.perf_counter() - start_time > timeout:
            return 1

        # check memory usage
        if memory_limit is None:
            continue

        memory_info = p.memory_info()  # get the memory usage of the process

        if memory_info.rss > memory_limit:
            p.terminate()
            return 2

        time.sleep(1e-1)  # reduce CPU load

    return -1


class AlignmentTool(Aligner):
    """ Wrapper for Nakamura's Alignment Tool.

    Links:
        https://midialignment.github.io/demo.html
        https://eita-nakamura.github.io/articles/EN_etal_ErrorDetectionAndRealignment_ISMIR2017.pdf
    """

    def __init__(
            self,
            tool_path: str,
            midi_dir: str,
            alignments_dir: str,
            sec_per_quarter_note: float = 1.,  # 0.001,
            extra_note_region: float = 0.3,  # ∆ in the paper
            check_err_match: bool = False,
            min_recall: float = 0.8
    ):
        self.tool_path = tool_path
        self._check_and_build_programs()

        self.midi_dir = midi_dir
        self.alignments_dir = alignments_dir
        self._build_file_directories()

        self.sec_per_quarter_note = sec_per_quarter_note
        self.extra_note_region = extra_note_region
        self.check_err_match = check_err_match
        self.min_recall = min_recall

    def _check_and_build_programs(self):
        # check all required computation programs
        self.programs_dir = os.path.join(self.tool_path, "Programs")
        assert os.path.exists(self.programs_dir), "No programs directory found, run `compile.sh`"

        self.programs = {}
        for program in PROGRAMS:
            program_path = os.path.join(self.programs_dir, program)
            assert os.path.exists(program_path), \
                f"Program {program} is not found under {self.programs_dir} directory"
            self.programs[program] = program_path

    def _build_file_directories(self):
        # build file directories under `alignments_dir`
        self.file_dirs = {}
        for name in FILE_DIRECTORIES:
            dirname = os.path.join(self.alignments_dir, name)
            os.makedirs(dirname, exist_ok=True)
            self.file_dirs[name] = dirname

    def align(
            self,
            score_midi: str | Path,
            perf_midi: str | Path,
            remove_intermediate_files: bool = True,
            force_recompute: bool = False,
            timeout: float | None = 1000.,
            memory_limit: int | None = 8_000_000_000,
            retries: int = 0
    ):
        start_time = time.perf_counter()

        paths = {DataType.SCORE.value: {}, DataType.PERFORMANCE.value: {}, DataType.ALIGNMENT.value: {}}
        error_messages = []

        # preprocess score midi
        score_spr, msg = self.midi_to_spr(score_midi, force_recompute=force_recompute)
        paths[DataType.SCORE][FileType.SPR.value] = score_spr
        error_messages.append(msg)

        score_fmt3x, msg = self.spr_to_fmt3x(score_spr, force_recompute=force_recompute)
        paths[DataType.SCORE][FileType.FMT3X.value] = score_spr
        error_messages.append(msg)

        score_hmm, msg = self.fmt3x_to_hmm(score_fmt3x, force_recompute=force_recompute)
        paths[DataType.SCORE][FileType.HMM.value] = score_spr
        error_messages.append(msg)

        # preprocess performance midi
        perf_spr, msg = self.midi_to_spr(perf_midi, force_recompute=force_recompute)
        paths[DataType.PERFORMANCE][FileType.SPR.value] = perf_spr
        error_messages.append(msg)

        # align files and return corresp file
        (match, corresp), msg = self.compute_alignment(
            score_spr=score_spr,
            score_fmt3x=score_fmt3x,
            score_hmm=score_hmm,
            perf_spr=perf_spr,
            remove_intermediate_files=remove_intermediate_files,
            force_recompute=force_recompute,
            retries=retries,
            timeout=timeout - (time.perf_counter() - start_time),  # time left
            memory_limit=memory_limit
        )
        paths[DataType.ALIGNMENT][FileType.MATCH.value] = match
        paths[DataType.ALIGNMENT][FileType.CORRESP.value] = corresp
        error_messages.append(msg)

        error_messages = list(filter(lambda m: len(m) > 0, error_messages))

        alignment = None
        if len(error_messages) == 0:
            alignment = Alignment(
                path=paths["align"]["corresp"],
                filetype="corresp"
            )

        return alignment, paths, error_messages

    def midi_to_spr(
            self,
            midi_path: str | Path,
            force_recompute: bool = False
    ):
        program = self.programs["midi2pianoroll"]
        midi_path = Path(midi_path)

        if not midi_path.exists():
            return None, f"Error@midi_to_spr: {FileType.MIDI} file not found ({midi_path})"

        spr_path = self.file_dirs[FileType.SPR] / midi_path.relative_to(self.midi_dir)
        spr_path = spr_path.parent / (spr_path.stem + FILE_EXT[FileType.SPR])
        out_path = midi_path.parent / (midi_path.stem + FILE_EXT[FileType.SPR])

        if not spr_path.exists() or force_recompute:
            spr_path.parent.mkdir(parents=True, exist_ok=True)  # process directories
            subprocess.run([program, "0", midi_path.with_suffix("")])  # run midi2pianoroll
            shutil.move(out_path, spr_path)  # move result to `spr_dir`

        if not spr_path.exists():
            return None, f"Error@midi_to_spr: computation of {FileType.SPR} file failed ({spr_path})"

        return spr_path, ""

    def spr_to_fmt3x(
            self,
            spr_path: str | Path,
            force_recompute: bool = False
    ):
        program = self.programs["SprToFmt3x"]
        spr_path = Path(spr_path)

        if not spr_path.exists():
            return None, f"Error@spr_to_fmt3x: {FileType.SPR} file not found ({spr_path})"

        fmt3x_path = self.file_dirs[FileType.FMT3X] / spr_path.relative_to(self.file_dirs[FileType.SPR])
        fmt3x_path = fmt3x_path.parent / fmt3x_path.name.replace(FILE_EXT[FileType.SPR], FILE_EXT[FileType.FMT3X])

        if not fmt3x_path.exists() or force_recompute:
            fmt3x_path.parent.mkdir(parents=True, exist_ok=True)  # process directories
            subprocess.run([program, spr_path, fmt3x_path])  # run SprToFmt3x

        if not fmt3x_path.exists():
            return None, f"Error@spr_to_fmt3x: computation of {FileType.FMT3X} file failed ({spr_path})"

        return fmt3x_path, ""

    def fmt3x_to_hmm(
            self,
            fmt3x_path: str | Path,
            force_recompute: bool = False
    ):
        program = self.programs["Fmt3xToHmm"]
        fmt3x_path = Path(fmt3x_path)

        if not fmt3x_path.exists():
            return None, f"Error@fmt3x_to_hmm: {FileType.FMT3X} file not found ({fmt3x_path})"

        hmm_path = self.file_dirs[FileType.HMM] / fmt3x_path.relative_to(self.file_dirs[FileType.FMT3X])
        hmm_path = hmm_path.parent / hmm_path.name.replace(FILE_EXT[FileType.FMT3X], FILE_EXT[FileType.HMM])

        if fmt3x_path.exists() and (not hmm_path.exists() or force_recompute):
            hmm_path.parent.mkdir(parents=True, exist_ok=True)  # process directories
            subprocess.run([program, fmt3x_path, hmm_path])  # run Fmt3xToHmm

        if not hmm_path.exists():
            return None, f"Error@fmt3x_to_hmm: computation of {FileType.HMM} file failed ({hmm_path})"

        return hmm_path, ""

    def compute_alignment(
            self,
            score_spr: str | Path,
            score_fmt3x: str | Path,
            score_hmm: str | Path,
            perf_spr: str | Path,
            remove_intermediate_files: bool = True,
            force_recompute: bool = False,
            timeout: float | None = 1000.,
            memory_limit: int | None = 8_000_000_000,
            retries: int = 0
    ):
        start_time = time.perf_counter()

        score_spr, score_fmt3x, score_hmm, perf_spr = map(
            lambda x: Path(x), (score_spr, score_fmt3x, score_hmm, perf_spr)
        )

        if any([not path.exists() for path in (score_spr, score_fmt3x, score_hmm, perf_spr)]):
            # essential files are missing, abort alignment tool
            return (None, None), "Error@compute_alignment: at least one of spr/fmt3x/hmm files missing"

        # prepare alignment name, directory and paths
        score, perf = map(lambda x: x.name.replace(FILE_EXT[FileType.SPR], ""), (score_spr, perf_spr))

        align_name = f"{perf}_{score}"
        match_dir = (self.file_dirs[FileType.MATCH] / perf_spr.relative_to(self.file_dirs[FileType.SPR])).parent
        match_dir.mkdir(parents=True, exist_ok=True)  # process directories

        match = match_dir / (align_name + FILE_EXT[FileType.MATCH])
        corresp = match_dir / (align_name + FILE_EXT[FileType.CORRESP])
        if match.exists() and corresp.exists() and not force_recompute:
            return (match, corresp), ""

        pre_match = match_dir / (align_name + FILE_EXT[FileType.PRE_MATCH])
        err_match = match_dir / (align_name + FILE_EXT[FileType.ERR_MATCH])
        err_corresp = match_dir / (align_name + FILE_EXT[FileType.ERR_CORRESP])

        def _delete_temporary_files():
            if remove_intermediate_files:
                if pre_match.exists():
                    os.remove(pre_match)
                if err_match.exists():
                    os.remove(err_match)
                if err_corresp.exists():
                    os.remove(err_corresp)

        def _process_status_error(file, filetype):
            if status == 0 and file.exists():
                return True, ""
            if status == 1:
                return False, f"TimeoutExpired@compute_alignment: computation of {filetype} " \
                              f"exceeded time limit of {timeout:.2f}s ({file})"
            elif status == 2:
                return False, f"MemoryError@compute_alignment: computation of {filetype} " \
                              f"exceeded memory limit of {memory_limit} bytes ({file})"
            return False, ""

        # compute pre-match
        if not pre_match.exists() or force_recompute:
            for i in range(retries + 1):
                status = run_subprocess_with_limits(
                    [self.programs["ScorePerfmMatcher"], score_hmm, perf_spr,
                     pre_match, str(self.sec_per_quarter_note)],
                    timeout=timeout - (time.perf_counter() - start_time),  # time left
                    memory_limit=memory_limit
                )

                succeed, error = _process_status_error(file=pre_match, filetype=FileType.PRE_MATCH)
                if succeed:
                    break
                elif len(error) > 0:
                    _delete_temporary_files()
                    return (None, None), error

        if not pre_match.exists():
            return (None, None), f"Error@compute_alignment: computation of {FileType.PRE_MATCH} failed ({pre_match})"

        # find error regions
        if not err_match.exists() or force_recompute:
            status = run_subprocess_with_limits(
                [self.programs["ErrorDetection"], score_fmt3x, score_hmm, pre_match, err_match, "0"],
                timeout=timeout - (time.perf_counter() - start_time),  # time left
                memory_limit=memory_limit
            )

            succeed, error = _process_status_error(file=err_match, filetype=FileType.ERR_MATCH)
            if len(error) > 0:
                _delete_temporary_files()
                return (None, None), error

        if not err_match.exists():
            return (None, None), f"Error@compute_alignment: computation of {FileType.ERR_MATCH} failed ({err_match})"

        if self.check_err_match:
            if not err_corresp.exists() or force_recompute:
                subprocess.run([self.programs["MatchToCorresp"], err_match, score_spr, err_corresp])

            err_alignment = Alignment.from_file(path=err_corresp)
            score_interval = err_alignment.end_index - err_alignment.start_index + 1
            recall = err_alignment.num_full_pairs / score_interval

            if recall < self.min_recall:
                _delete_temporary_files()
                return (None, None), f"LowMatchRatio@compute_alignment: " \
                                     f"computation of {FileType.MATCH} halted ({match}), " \
                                     f"{FileType.ERR_CORRESP}'s match ratio: " \
                                     f"{recall:.3f} < {self.min_recall:.3f}"

        # realignment and match file
        if not match.exists() or force_recompute:
            for i in range(retries + 1):
                status = run_subprocess_with_limits(
                    [self.programs["RealignmentMOHMM"], score_fmt3x, score_hmm,
                     err_match, match, str(self.extra_note_region)],
                    timeout=timeout - (time.perf_counter() - start_time),  # time left
                    memory_limit=memory_limit
                )

                succeed, error = _process_status_error(file=match, filetype=FileType.MATCH)
                if succeed:
                    break
                elif len(error) > 0:
                    _delete_temporary_files()
                    return (None, None), error

        _delete_temporary_files()

        if not match.exists():
            return (None, None), f"Error@compute_alignment: computation of {FileType.MATCH} failed ({match})"

        # transform match to corresp
        subprocess.run([self.programs["MatchToCorresp"], match, score_spr, corresp])

        if not corresp.exists():
            return (match, None), f"Error@compute_alignment: computation of {FileType.CORRESP} failed ({corresp})"

        return (match, corresp), ""
