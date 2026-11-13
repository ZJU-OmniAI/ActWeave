import fcntl
from pathlib import Path
import time


def reserve_gpu_memory(device, memory, domain, selector, deadline):
    import torch
    Path(domain).mkdir(parents=True, exist_ok=True)
    with (Path(domain) / ('gpu-' + str(selector) + '-admission.lock')).open('a+') as admission:
        while time.time() < deadline:
            try:
                fcntl.flock(admission, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                time.sleep(0.1)
                continue
            try:
                free, total = torch.cuda.mem_get_info(device)
                if free >= memory['min_free_gib'] * 1024 ** 3:
                    reservation = torch.empty(int(memory['startup_reserve_gib'] * 1024 ** 3), dtype=torch.uint8, device=device)
                    reservation.fill_(0)
                    return reservation
            finally:
                fcntl.flock(admission, fcntl.LOCK_UN)
            time.sleep(min(1, max(0, deadline - time.time())))
    raise TimeoutError('GPU memory admission deadline')

