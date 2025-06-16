mkdir data
cd data

# SMALL Scale
# LSUN-C dataset
wget https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz
tar -xvzf LSUN.tar.gz && rm -f LSUN.tar.gz

# LSUN-R dataset
wget https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz
tar -xvzf LSUN_resize.tar.gz && rm -f LSUN_resize.tar.gz

# Texture dataset
wget https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz
tar -xvzf dtd-r1.0.1.tar.gz && rm -f dtd-r1.0.1.tar.gz

# Places365 dataset (small scale)
wget http://data.csail.mit.edu/places/places365/test_256.tar
tar -xvzf test_256.tar && rm -f test_256.tar

# iSUN dataset
wget https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz
tar -xvzf iSUN.tar.gz && rm -f iSUN.tar.gz

# LARGE Scale
# iNaturalist dataset
# wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/iNaturalist.tar.gz
# tar -xvzf iNaturalist.tar.gz && rm -f iNaturalist.tar.gz

# SUN dataset
# wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/SUN.tar.gz
# tar -xvzf SUN.tar.gz && rm -f SUN.tar.gz

# Places365 dataset (large scale)
# wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/Places.tar.gz
# tar -xvzf Places.tar.gz && rm -f Places.tar.gz