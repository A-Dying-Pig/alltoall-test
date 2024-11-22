target_node="02"
declare -a arr=('11' '12' '21' '22' '31' '32' '41' '42')
for tar in "${arr[@]}"
do
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