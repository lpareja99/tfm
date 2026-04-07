Mount SSD for Linux

1. Powersell: `$ wsl --mount \\.\PHYSICALDRIVE1 --partition 1`

2. Ubuntu: 
`lsblk -f` Copy UU_ID
`sudo mount -U b126e038-8646-4b24-951f-acf6d6d1dd82 /home/lpa/external_master_ssd`
`sudo chown -R $USER:$USER ~/external_master_ssd`+
`sudo chown -R lpa:lpa /home/lpa/external_master_ssd`
`df -h /home/lpa/external_master_ssd`


Dismount SSD for Linux

1. Ubuntu `sudo umount ~/external_master_ssd`
2. PowerShell: `wsl --unmount \\.\PHYSICALDRIVE1`

or at least powershell: `wsl --shutdown`
