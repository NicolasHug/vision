#!/bin/bash
archive_size=1000

for split in "val" "train"
do
    for archiver in "torch" "pickle"
    do
        for archive_content in "tensor" "bytesio"
        do
            python make_archives.py --input-dir /ontap_isolated/nicolashug/tinyimagenet/081318/$split --output-dir /fsx_isolated/nicolashug/tinyimagenet/081318/archives/$split --archiver $archiver --archive-content $archive_content --archive-size $archive_size

        done
    done
    python make_archives.py --input-dir /ontap_isolated/nicolashug/tinyimagenet/081318/$split --output-dir /fsx_isolated/nicolashug/tinyimagenet/081318/archives/$split --archiver tar --archive-size $archive_size
done