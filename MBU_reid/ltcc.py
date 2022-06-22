import glob
import re
import pdb
import os.path as osp
import os
from .bases import ImageDataset
import warnings



from fastreid.data.datasets import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class LTCC(ImageDataset):
    dataset_dir = 'LTCC'
    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(LTCC, self).__init__()
        self.pid_begin = pid_begin
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')

        self._check_before_run()

        train = self.process_dir(self.train_dir, relabel=True)
        query = self.process_dir(self.query_dir, relabel=False)
        gallery = self.process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> ltcc loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(
            self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(
            self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(
            self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def process_dir(self, dir_path, mode='train', relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*g'))
        pattern = re.compile(r'(.*\d.*)_(.*\d.*)_c(.*\d.*)_(.*\d.*)')


        pid_container = set()
        for img_path in img_paths:
            pid, _, _, _ = pattern.search(img_path).groups()
            pid = pid.split('/')[-1]
            pid_container.add(pid)
            # print("pid", pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        # print("img paths", img_paths)
        for img_path in img_paths:
            pids, _, camid, _ = pattern.search(img_path).groups()
            # print("pid", pids, camid)
            pids = pids.split('/')[-1]
            # print("pid1", pids, camid)

            pid = pids

            # print("relabel", relabel)

            if relabel:
                pid = pid2label[pids]
            else:
                pids = pids.split('\\')[-1]
                # print("pids", pids)
                # print("pids", pids.startswith('b'))
                if pids.startswith('b'):
                    pid = pids.split('_')[1]
                else:
                    pid = pids
                    # pid = pids.split('\\')[-1]
            if pids.startswith('b'):
                black_id = 1
            else:
                black_id = 0
            # print("pids", pid)
            pid = int(pid)
            camid = int(camid)
            black_id = int(black_id)

            if mode == 'train':
                data.append((img_path, pid, camid, black_id))
            else:
                data.append((img_path, pid, camid))

        return data
