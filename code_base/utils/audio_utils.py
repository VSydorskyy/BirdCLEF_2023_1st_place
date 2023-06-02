from copy import deepcopy

import librosa

try:
    import noisereduce as nr
except:
    print("`noisereduce` was not imported")
from joblib import delayed

from .main_utils import ProgressParallel


def get_librosa_load(
    do_normalize,
    do_noisereduce=False,
    pos_dtype=None,
    return_au_len=False,
    **kwargs,
):
    def librosa_load(path):
        # assert kwargs["sr"] == 32_000
        try:
            au, sr = librosa.load(path, **kwargs)
            if do_noisereduce:
                try:
                    au = nr.reduce_noise(y=deepcopy(au), sr=sr)
                    if do_normalize:
                        au = librosa.util.normalize(au)
                    return au, sr
                except Exception as e:
                    print(f"{e} was catched while `reduce_noise`")
                    au, sr = librosa.load(path, **kwargs)
            if do_normalize:
                au = librosa.util.normalize(au)
            if pos_dtype is not None:
                au = au.astype(pos_dtype)
            if return_au_len:
                au = len(au)
            return au, sr
        except Exception as e:
            print("librosa_load failed with {e}")
            return None, None

    return librosa_load


def load_pp_audio(
    name,
    sr=None,
    normalize=True,
    do_noisereduce=False,
    pos_dtype=None,
    res_type="kaiser_best",
    validate_sr=None,
):
    # assert sr == 32_000
    au, sr = librosa.load(name, sr=sr)
    if validate_sr is not None:
        assert sr == validate_sr
    if do_noisereduce:
        try:
            au = nr.reduce_noise(y=deepcopy(au), sr=sr, res_type=res_type)
            if normalize:
                au = librosa.util.normalize(au)
            return au
        except Exception as e:
            print(f"{e} was catched while `reduce_noise`")
            au, sr = librosa.load(name, sr=sr)
    if normalize:
        au = librosa.util.normalize(au)
    if pos_dtype is not None:
        au = au.astype(pos_dtype)
    return au


def parallel_librosa_load(
    audio_pathes,
    n_cores=32,
    return_sr=True,
    return_audio=True,
    do_normalize=False,
    **kwargs,
):
    assert return_sr or return_audio
    complete_out = ProgressParallel(n_jobs=n_cores, total=len(audio_pathes))(
        delayed(get_librosa_load(do_normalize=do_normalize, **kwargs))(el_path)
        for el_path in audio_pathes
    )
    if return_sr and return_audio:
        return complete_out
    elif return_audio:
        return [el[0] for el in complete_out]
    elif return_sr:
        return [el[1] for el in complete_out]
