from queue import Queue, Empty
from threading import Thread
import subprocess
import sys
import time

def custom_async(f):
    def wrapper(*args, **kwargs):
        thr = Thread(target=f, args=args, kwargs=kwargs)
        thr.start()
    return wrapper

class process_pool(object):

    def __init__(self, num_buffer_lines=5):
        # self.num_buffer_lines = num_buffer_lines
        # self.cur = 0
        self.reset()

    def reset(self):
        self.processes = []
        self.message_queue = Queue()
        self.activated = False

    def start(self, cmd, idx, cwd):
        p = subprocess.Popen(cmd,  
                            shell=True, 
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.STDOUT,
                            cwd=cwd,
                            encoding='utf-8'
                            )
        self.activated |= True
        t = Thread(target=self.enqueue_output, args=(p.stdout, idx))
        t.daemon=True
        t.start()
        self.processes.append((idx, p, t))

    def apply(self, cmd_cwd_list):
        for idx, cmd_cwd in enumerate(cmd_cwd_list):
            cmd, cwd = cmd_cwd
            self.start(cmd, idx, cwd)
        self.daemon()

    def enqueue_output(self, out, i):
        for line in iter(out.readline, b''):
            self.message_queue.put_nowait((i, line))
        out.close()

    def create_buffers(self):
        self.all_num_buffer_lines = self.num_buffer_lines*self.process_num
        self.buffer_start_list = list(range(0, self.all_num_buffer_lines, self.num_buffer_lines))
        self.buffer_cur_list = list(range(0, self.all_num_buffer_lines, self.num_buffer_lines))
        sys.stdout.write('\n'*self.all_num_buffer_lines)
        self.cur += self.all_num_buffer_lines

    @custom_async
    def daemon(self):
        self.process_num = len(self.processes)
        # self.create_buffers()
        alive_pool = [1 for _ in range(self.process_num)]
        
        while True:
            if sum(alive_pool) == 0:
                break
            try: 
                i, out = self.message_queue.get_nowait()
            except Empty:
                pass
            else:
                # to_print_cur = self.buffer_cur_list[i] + 1
                # if to_print_cur == self.buffer_start_list[i] + self.num_buffer_lines:
                #     to_print_cur = self.buffer_start_list[i]
                
                # if self.cur > to_print_cur:
                #     up_offset = self.cur - to_print_cur
                #     sys.stdout.write('\x1b[%dA'%up_offset)
                # elif self.cur < to_print_cur:
                #     down_offset = to_print_cur - self.cur
                #     sys.stdout.write('\x1b[%dB'%up_offset)
                # else:
                #     pass
                # self.cur = to_print_cur
                # sys.stdout.write(out.strip())

                out_strip = out.replace('\x1b[A','\n').strip()
                if len(out_strip) > 0:
                    if self.process_num > 1:
                        sys.stdout.write(' '.join(['pid: {:d}'.format(i), out_strip]))
                    else:
                        sys.stdout.write(out_strip)
                    sys.stdout.flush()
                    sys.stdout.write('\n')
                    sys.stdout.flush()
            for pid, p, _, in self.processes:
                if p.poll() is not None:
                    alive_pool[pid] = 0
        self.reset()
    
    def wait(self):
        while True:
            if not self.activated:
                break
            else:
                pass
                # time.sleep(0.1)