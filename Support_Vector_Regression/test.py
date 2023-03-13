import neo


r = neo.io.Spike2IO(filename='rgb20190528_180044_6684_json/20190528_180044_6684.hsd')
img = r.read_block()
print(img)
