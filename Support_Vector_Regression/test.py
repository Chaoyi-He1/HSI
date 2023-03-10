import neo


r = neo.io.Spike2IO(filename='/path/to/file.hsd')
img = r.read_block()
