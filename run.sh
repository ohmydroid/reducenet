for dataset in cifar10 cifar100  
do
	  for model in reduce20 reduce56 
	  do
		   for ws in 1 2 4
		   do
		     python main.py -d $dataset -m $model --seed $seed -ws $ws
		   done
	  done
done 
