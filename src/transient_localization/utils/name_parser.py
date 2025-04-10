class NameParser:
    def __init__(self, filename="CIGRELVDist_train_step_0Wrnd-0-0.1.csv"):
        underscore_split = filename[:-4].split("-")
        self.transient_type = underscore_split[0].split("_")[-1]
        self.bus = int(underscore_split[1])
        self.strength = float(underscore_split[2])
