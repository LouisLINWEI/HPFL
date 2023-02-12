### set home directory
home=$HOME'/wlin/HPFL'

master='localhost'
# workers='localhost localhost localhost localhost localhost localhost localhost localhost localhost localhost localhost localhost localhost localhost localhost localhost localhost localhost localhost localhost localhost localhost localhost localhost localhost localhost localhost localhost localhost localhost localhost localhost'
workers='localhost localhost localhost localhost'

echo 'master(coordinator): '$master
echo 'worker_hosts: '$workers
world_size=0
for i in $workers
do
		world_size=$((world_size+1))
done
### create log dir and code snapshot
read -p "enter exp-setup notes: " remarks
read -p "Specify allocated GPU-ID (world_size: $world_size): " cuda 
trial_no=$(ls $home/Logs/ | wc -l)
port_no=$((20000+trial_no))
log_dir=$home/Logs/${trial_no}_$remarks

mkdir -p $log_dir/code_snapshot 
cp $home/train.sh $log_dir/code_snapshot
cp $home/*.py $log_dir/code_snapshot
cp -r $home/models $log_dir/code_snapshot
rm -f Latest_Log && ln -s $log_dir Latest_Log


### launch process
num=0
job=worker_process_measure.py
job=worker_process_as.py
for i in $workers
do
	command="source /root/miniconda3/bin/activate && conda source activate flamby && export CUDA_VISIBLE_DEVICES=0,1,2,3 && python3 $home/$job --master_address=tcp://${master}:$port_no --rank=$num --world_size=$world_size --remarks=$remarks --trial_no=$trial_no"
	# echo $command
	nohup ssh $i $command >$log_dir/worker_$num.log 2>&1 &
	num=$((num+1))
done
echo $num' worker processes have been launched!'
echo 'Please check logs in path: '$log_dir
