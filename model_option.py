import argparse



class Parser(object):
    def __init__(self, decript):
        self.argments = argparse.ArgumentParser(description = decript)

    def add(self, type, help, arg, default):
        return self.argments.add_argument(
            "".join(["--", arg]),
            default = default,
            type = type,
            help = help)
    
    def parsing(self):
        return self.argments.parse_args()



def options():
    parser = Parser(decript = "Input optional guidance for training")
    parser.add(str, "데이터셋 경로", 
        "datapath", "/SSD/kitti")
    parser.add(str, ["kitti", "eigen"], 
        "splits", "kitti")

    parser.add(int, "에포크",
        "epoch", 70)
    parser.add(int, "배치 사이즈", 
        "batch", 8)
    parser.add(int, "prefetch_factor", 
        "prepetch", 2)
    parser.add(int, "num_workers", 
        "num_workers", 16)
    parser.add(float, "learning_rate", 
        "learning_rate", 0.0002)
    parser.add(str, "save file name", 
        "save", "custom_kitti")
    parser.add(int, "save freq rate", 
        "freq", 2)

    parser.add(int, "이미지의 높이",
        "height", 256)
    parser.add(int, "이미지의 너비",
        "width", 832)
    parser.add(str, "이미지 확장자",
        "img_ext", ".jpg")
    return parser.parsing()