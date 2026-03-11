#!/usr/bin/expect -f
# Usage: ./vps_scp.sh local_file remote_destination_path
# Example: ./vps_scp.sh config.yaml ~/cnn_lstm_v1/config.yaml
set timeout 60
set local_file [lindex $argv 0]
set remote_path [lindex $argv 1]
spawn scp -o StrictHostKeyChecking=no $local_file root@143.198.133.0:$remote_path
expect {
    "password:" { send "7h-Pq2NmXw9K\r"; exp_continue }
    eof
}
catch wait result
exit [lindex $result 3]
