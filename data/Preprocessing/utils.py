import numpy as np
import binascii
import scapy.all as scapy


# hex to 0-255
def string_to_hex_array(flow_string):
    return np.array([int(flow_string[i:i + 2], 16) for i in range(0, len(flow_string), 2)])


def read_pcap_list(pcap_filename, if_augment=False, remove_ip=True, keep_payload=True):
    """
    32*32
    """
    n_1, n_2 = 160, 96
    packets = scapy.rdpcap(pcap_filename)
    data = []
    flow_string_length = 2048
    flow_packet_num = 8
    end = len(packets) if if_augment else flow_packet_num
    for packet in packets[:end]:
        try:
            header, payload = raw_packet_to_string(packet, remove_ip=remove_ip, keep_payload=keep_payload)
        except:
            header, payload = '0' * n_1, '0' * n_2
        data.append(header + payload)

    if not if_augment or len(data) <= flow_packet_num:
        flow_string = ''.join(data)
        flow_string += '0' * (flow_string_length - len(flow_string))
        flow_array = string_to_hex_array(flow_string)
        return [{
            "data": flow_array,
        }]
    else:
        assert len(data) > flow_packet_num
        flow_array_list = []
        for i in range(len(data) - flow_packet_num + 1):
            flow_string = ''.join(data[i:i + flow_packet_num])
            flow_array_list.append(string_to_hex_array(flow_string))
        return [{
            "data": flow_array,
        } for flow_array in 
        zip(flow_array_list)]


def raw_packet_to_string(packet, remove_ip=True, keep_payload=True):
    """
    n_1, n_2 = 160, 96
    """
    n_1, n_2 = 160, 96
    ip = packet["IP"]
    if remove_ip:
        PAD_IP_ADDR = "0.0.0.0"
        ip.src, ip.dst = PAD_IP_ADDR, PAD_IP_ADDR
    header = (binascii.hexlify(bytes(ip))).decode()
    if keep_payload:
        try:
            payload = (binascii.hexlify(bytes(packet['Raw']))).decode()
            header = header.replace(payload, '')
        except:
            payload = ''
    else:
        payload = ''
    header = header[:n_1] if len(header) > n_1 else header + '0' * (n_1 - len(header))
    payload = payload[:n_2] if len(payload) > n_2 else payload + '0' * (n_2 - len(payload))
    return header, payload



if __name__ == "__main__":
    pass

