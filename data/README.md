# Dataset
Uses the SceneFlow FlythinThings3D and Driving datasets. Specifically, the compressed finalpass RGB images and the disparity maps. These can be downloaded from [academictorrents](https://academictorrents.com/userdetails.php?id=9551). 

The directory structure needs to look like the following, where `[ A | B | C]` refers to the three directories A, B, C and `xxxx` refers to the 4 digit subdirectory names.
```
sceneflow
├── train
│   ├── disparity
│   │   └── [ A | B | C ]
│   │       └── xxxx
│   └── images
│       └── [ A | B | C ]
│           └── xxxx
│               ├── left
│               └── right
├── val
│   ├── disparity
│   │   └── [ A | B | C ]
│   │       └── xxxx
│   └── images
│       └── [ A | B | C ]
│           └── xxxx
│               ├── left
│               └── right
├── test_15mm
│   ├── disparity
│   └── images
│       ├── left
│       └── right
└── test_35mm
    └── disparity
    └── images
        ├── left
        └── right
```

The dataset gets quite large (easily 200 GB unpruned), so the filenames are dumped to files in this directory to avoid spending ~1/2 minute traversing the entire tree. This should happen automatically, but can be done manually with: `cache_paths.py`