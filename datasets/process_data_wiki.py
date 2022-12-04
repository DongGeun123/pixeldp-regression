import os
import numpy as np
import utils

def main():
    rootpath="./wiki"
    outfile="./wiki_list.txt"
    metafile ="wiki"
    min_score = 1.0
    full_path, dob, gender, photo_taken, face_score, second_face_score, age=utils.get_meta(os.path.join(rootpath,'%s.mat'%metafile),metafile)

    total = 0

    label = []
    print("%d images " % len(face_score))
    for i in range(len(face_score)):
        if face_score[i] < min_score:
            continue

        if (~np.isnan(second_face_score[i])) and second_face_score[i] > 0.0:
            continue

        if ~(0 <= age[i] <= 100):
            continue

        if np.isnan(gender[i]):
            continue
        fname=str(full_path[i][0])

        label.append([fname, age[i], gender[i]])
        total +=1

    with open(os.path.join(rootpath,outfile),'w') as f:
        for  line in label:
            f.write(line[0] + '\t' + str(line[1]) + '\t' + str(line[2]) + '\n')
    print("filter data")
    print("total: %d image" %(total))
    print('Done!!!')


if __name__ =='__main__':
    main()
