
# CONFIGURATION, Window Definitions & MAPPINGS
TARGET_SR = 20  # Hz
CHANNELS = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'] # 6-channel schema 

# Windowing Definitions
PRETRAIN_WINDOW_SEC = 10
PRETRAIN_OVERLAP = 0.0
SUPERVISED_WINDOW_SEC = 5
SUPERVISED_OVERLAP = 0.5

# Shared Label Schema Mapping
# Unified 5-Class Schema:
'''{ 1 = Sitting
     2 = Standing
     3 = Walking
     4 = Running / Jogging (Merged)
     5 = Stairs (Merged)
     }'''

LABEL_MAP_PAMAP2 = {
    2: 1,   # sitting -> Sitting
    3: 2,   # standing -> Standing
    4: 3,   # walking -> Walking
    5: 4,   # running -> Running/Jogging
    12: 5,  # ascending stairs -> Stairs
    13: 5   # descending stairs -> Stairs
}

LABEL_MAP_WISDM = {
    'D': 1,  # Sitting -> Sitting
    'E': 2,  # Standing -> Standing
    'A': 3,  # Walking -> Walking
    'B': 4,  # Jogging -> Running/Jogging
    'C': 5   # Stairs -> Stairs
}

LABEL_MAP_MHEALTH = {
    2: 1,   # sitting still -> Sitting
    1: 2,   # standing still -> Standing
    4: 3,   # walking -> Walking
    10: 4,  # jogging -> Running/Jogging
    11: 4,  # running -> Running/Jogging
    5: 5    # climbing -> Stairs
}

