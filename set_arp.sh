target_node="03"
this_node="02"
declare -a arr=('11' '12' '21' '22' '31' '32' '41' '42')
for tar in "${arr[@]}"
do
    interface="ens${tar}f0np0"
    if_cmd_down="ifconfig ${interface} down"
    ip="10.0.${tar}.${this_node}"
    if_cmd_mac="ifconfig ${interface} hw ether 08:0a:35:9e:${tar}:${this_node}"
    if_cmd_up="ifconfig ${interface} ${ip}/24 up"
    echo ${if_cmd_down}
    echo ${if_cmd_mac}
    echo ${if_cmd_up}
    eval ${if_cmd_down}
    eval ${if_cmd_mac}
    eval ${if_cmd_up}
    for inf in "${arr[@]}"
    do
        mac="08:0a:35:9e:${inf}:${target_node}"
        ip="10.0.${inf}.${target_node}"
        # target_inf="ens${tar}np0"
        target_inf="ens${tar}f0np0"
        del_cmd="arp -d ${ip} -i ${target_inf}"
        cmd="arp -s ${ip} ${mac} -i ${target_inf}"
        echo ${cmd}
        echo ${del_cmd}
        eval ${del_cmd}
        eval ${cmd}
    done
done