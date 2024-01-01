import time
local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

class opts():
    def __init__(self):
        self.input_C = 4
        self.input_H = 240
        self.input_W = 240
        self.input_D = 160
        self.output_D = 155

        self.crop_H = 128
        self.crop_W = 128
        self.crop_D = 128

    def gatherAttrs(self):
        return ",".join("\n{}={}"
                        .format(k, getattr(self, k))
                        for k in self.__dict__.keys())

    def __str__(self):
        return "{}:{}".format(self.__class__.__name__, self.gatherAttrs())



