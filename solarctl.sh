#/bin/bash

INSTANCE_ID="i-04762a497fb52de9f"
STATUS_QUERY="Reservations[*].Instances[*].{InstanceId:InstanceId,State:State,Tags:Tags[*]}"
OPTIONS="[status|start|stop|restart|reboot|help|info]"

case $1 in
"" | help | info )
    cat << END
Manage the solar EC2 instance.
Usage: $0 $OPTIONS
END
    ;;
status )
    aws ec2 describe-instances --instance-ids $INSTANCE_ID --query $STATUS_QUERY
    ;;
start )
    aws ec2 start-instances --instance-ids $INSTANCE_ID
    ;;
stop )
    aws ec2 stop-instances --instance-ids $INSTANCE_ID
    ;;
restart | reboot )
    aws ec2 reboot-instances --instance-ids $INSTANCE_ID
    ;;
* )
    echo "Error: expected one of $OPTIONS; got $1"
    ;;
esac
