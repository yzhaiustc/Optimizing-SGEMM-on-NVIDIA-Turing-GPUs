rm res*
for i in {0..11..1}
do
	file_name="res_${i}.txt"
	./sgemm_gpu $i >> ${file_name}
done
