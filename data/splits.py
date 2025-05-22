# Scenario A-1, A-2, A-3
USTC = {
    'known_set': 'USTC',
    'unknown_set': 'USTC',
    'splits': [
        {'known_classes': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19], 'unknown_classes': [16]},
        {'known_classes': [0,1,3,4,7,8,9,11,14,17, 2,5,6,10,12,18,19], 'unknown_classes': [13,15,16]},
        {'known_classes': [0,1,3,4,7,8,9,11,14,17, 2,5,6,10,12], 'unknown_classes': [13,15,16,18,19]},
    ]
}
# {'Gmail': 0, 'FTP': 1, 'Nsis-ay': 2, 'Facetime': 3, 'Weibo': 4, 'Cridex': 5, 'Zeus': 6, 'SMB': 7, 'BitTorrent': 8, 'WorldOfWarcraft': 9, 'Shifu': 10, 
# 'Outlook': 11, 'Virut': 12, 'Geodo': 13, 'MySQL': 14, 'Htbot': 15, 'Tinba': 16, 'Skype': 17, 'Miuref': 18, 'Neris': 19}

# Scenario B-1, B-2, B-3
mal = {
    'known_set': 'mal',
    'unknown_set': 'mal',
    'splits': [
        {'known_classes': [0, 1, 2, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20, 21, 22, 23], 'unknown_classes': [3]},
        {'known_classes': [3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20, 21, 22, 23], 'unknown_classes': [0, 1, 2]},
        {'known_classes': [5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20, 21, 22, 23], 'unknown_classes': [0, 1, 2, 3, 4]},
    ]
}


# mal的标签是0-23之间，USTC的标签在24-43之间，新的数据标签在0-43，共44类数据
# Scenario C-1, C-2
combined_USTC_mal = {
    'known_set': 'combined_USTC_mal',
    'unknown_set': 'combined_USTC_mal',
    'splits': [
        {'known_classes': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], 'unknown_classes': [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43]},
        {'known_classes': [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43], 'unknown_classes': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]},
    ]
}

datasets = {
    'mal': mal,
    'USTC': USTC,
    'combined_USTC_mal': combined_USTC_mal
}

def get_splits(dataset, num_split):
    known_classes = datasets[dataset]['splits'][num_split]['known_classes']
    unknown_classes = datasets[dataset]['splits'][num_split]['unknown_classes']
    known_dataset = datasets[dataset]['known_set']
    unknown_dataset = datasets[dataset]['unknown_set']

    return known_classes, unknown_classes, known_dataset, unknown_dataset



