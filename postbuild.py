import os
examples = [x[0] for x in os.walk('./examples') if x[0] != './examples']
#print(examples)
src = os.path.abspath("./lightgbm")
for example in examples:
    #print(example)
    target = '%s/lightgbm' % example
    print("linking ./lightgbm -> %s" % target)
    if os.path.isfile(target):
        os.remove(target)
        pass
    os.symlink(src,target)
    pass
