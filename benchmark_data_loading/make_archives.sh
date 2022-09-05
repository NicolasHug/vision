#!/bin/bash
archive_size=500
input_dir=/ontap_isolated/nicolashug/imagenet_full_size/061417/train
output_dir=/ontap_isolated/nicolashug/imagenet_full_size/061417/archives/train

for archiver in "torch" "pickle"
do
    for archive_content in "tensor" "bytesio" "decoded"
    do
        python make_archives.py --input-dir $input_dir --output-path $output_dir --archiver $archiver --archive-content $archive_content --archive-size $archive_size
    done
done

python make_archives.py --input-dir $input_dir --output-path $output_dir --archiver tar --archive-size $archive_size

for archive_content in "" "decoded"
do
    output_path="${output_dir}/ffcv"
    if [[ -n $archive_content ]]; then
       output_path="${output_path}_${archive_content}"
    fi
    output_path="${output_path}.beton"
    python make_archives.py --input-dir $input_dir --output-path $output_path --archiver ffcv --archive-content=$archive_content
done
