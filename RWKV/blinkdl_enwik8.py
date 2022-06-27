import datetime, sys, time
import vast, logging, fabric, tqdm
logging.basicConfig(level=logging.INFO)
#logging.basicConfig(level=logging.DEBUG)
#logging.getLogger("paramiko").setLevel(level=logging.DEBUG)

#instance = vast.Instance(query='', sort='flops_usd-', instance_type='on-demand', image='nvidia/cuda:11.3.1-devel-ubuntu20.04', GiB=8)
instance = vast.Instance(query='total_flops>30', sort='flops_usd-', instance_type='on-demand', image='pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel', GiB=8)
instance.create()
try:
    # at 43 TFlops, each of 500 epochs was taking about 3:20
    expected_seconds = (3*60+20) * 500 * 43 / instance.total_flops
    expected_hours = expected_seconds / 60 / 60
    expected_price = instance.dph_total * expected_hours
    print('Machine: ', instance.machine_id, ' Flops: ', instance.total_flops, ' Price/hr: $', instance.dph_total)
    print('Expected runtime: ', str(datetime.timedelta(seconds=expected_seconds)))
    print('Expected price: $', int(1+expected_price*100)/100)
    instance.wait()
    #instance.copy('enwik8', 'enwik8')
    print(f'Connecting to {instance.ssh_host}:{instance.ssh_port}')
    with fabric.Connection(instance.ssh_host, 'root', instance.ssh_port) as shell:
        # if asynchronous=True is passed to run, output is hidden and a promise is returned with a .join() method, to run tasks in parallel

        shell.run('nvidia-smi')

        #shell.shell() # opens interactive shell and takes many kwparams if needed

        shell.run('apt-get install -y ninja-build')
        shell.run('git clone https://github.com/BlinkDL/RWKV-LM')

        with tqdm.tqdm(desc='enwik8', unit='iB', unit_scale=True, unit_divisor=1024) as pbar:
            def progress(amt, tot):
                pbar.total = tot
                pbar.update(amt - pbar.n)
            sftp = shell.client.open_sftp()
            sftp.put('enwik8', 'RWKV-LM/RWKV-v2-RNN/enwik8', callback=progress)
        sftp.close()


        shell.run('echo PS1=$ . .bashrc > enter')
        shell.run('echo cd RWKV-LM/RWKV-v2-RNN >> enter')

        # compile cuda kernel
        shell.run(". enter; python3 -c 'import src.model'", pty=True)

        print()
        print('Here we go. ETA:', datetime.datetime.fromtimestamp(time.time() + expected_seconds).isoformat())
        print()

        shell.run(". enter; python3 -c 'import importlib.abc, runpy; runpy._run_module_as_main(\"train\")'", pty=True)
    
finally:
    instance.destroy()

