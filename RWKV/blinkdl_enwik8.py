import datetime, io, sys, time
import vast, logging, fabric, tqdm
logging.basicConfig(level=logging.INFO)

#instance = vast.Instance(query='gpu_ram>8.58', sort='flops_usd-', instance_type='on-demand', image='nvidia/cuda:11.3.1-devel-ubuntu20.04', GiB=2)
#instance = vast.Instance(query='gpu_ram>8.58 total_flops>30', sort='flops_usd-', instance_type='on-demand', image='pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel', GiB=2)
instance = vast.Instance(query='gpu_ram>8.58', sort='dph_total', instance_type='on-demand', image='pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel', GiB=2)
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
    print(f'System available as: ssh -p {instance.ssh_port} root@{instance.ssh_host}')

    with fabric.Connection(instance.ssh_host, 'root', instance.ssh_port) as shell:
        # if asynchronous=True is passed to run, output is hidden and a promise is returned with a .join() method, to run tasks in parallel

        shell.run('nvidia-smi')

        # open interactive shell, many kwparams if needed
        #shell.shell()
        #shell.run('bash', pty=True)

        shell.run('apt-get install -y ninja-build')
        shell.run('git clone --branch importlib_abc_workaround https://github.com/xloem/RWKV-LM')

        sftp = shell.client.open_sftp()
        with tqdm.tqdm(desc='enwik8', unit='iB', unit_scale=True, unit_divisor=1024) as pbar:
            def progress(amt, tot):
                pbar.total = tot
                pbar.update(amt - pbar.n)
            sftp.put('enwik8', 'RWKV-LM/RWKV-v2-RNN/enwik8', callback=progress)

        sftp.putfo(io.StringIO('''
                PS1=$ # bashrc adds conda to path if PS1 set
                . .bashrc
                cd RWKV-LM/RWKV-v2-RNN
                "$@"
            '''),
            'in_tree_run')
        shell.run('chmod 755 in_tree_run')

        sftp.close()

        shell.run('touch .no_auto_tmux')

        # compile cuda kernel
        shell.run('./in_tree_run python3 src/model.py')

        print()
        print('Here we go. ETA:', datetime.datetime.fromtimestamp(time.time() + expected_seconds).isoformat())
        print()
        #import pdb; pdb.set_trace()

        date_start = datetime.datetime.now()

        shell.run('tmux new-session -d -s job "./in_tree_run python3 train.py"', pty=True)

        shell.run('tmux attach-session -t job', pty=True)

        files = sftp.listdir('RWKV-LM/RWKV-v2-RNN')
        
        sftp.get('RWKV-LM/RWKV-v2-RNN/mylog.txt', f'blinkdl_enwik8-{date_start.isoformat()}.txt')
        for file in files:
            if file.endswith('.pth'):
                sftp.get('RWKV-LM/RWKV-v2-RNN/' + file, f'blinkdl_wneik8-{date_start.isoformat()}-{file}')
    
finally:
    instance.destroy()

