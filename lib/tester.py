from lib.trainer import Trainer
import torch
from tqdm import tqdm
from models.loss import MatchMotionLoss as MML
import numpy as np
from models.matching import Matching as CM
import math
import open3d as o3d
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime


class _3DMatchTester(Trainer):
    """
    3DMatch tester
    """
    def __init__(self,args):
        Trainer.__init__(self, args)
        expt_dir = time.strftime("%D_%H-%M", time.localtime(time.time())).replace('/','-')
        self.results_dir = os.path.join(self.config.results_dir, expt_dir)
        os.makedirs(self.results_dir)
        self.log_file = open(os.path.join(self.results_dir, "3d_match_results.txt"), "w+")

    def test(self):
        n = 10

        # afmr = 0.
        # arr = 0
        # air = 0

        for i in range(n): # combat ransac nondeterministic

            thr =0.05
            self.test_thr(i, thr)

        self.log_file.close()
            # rr, ir, fmr = self.test_thr(thr)
        #     afmr+=fmr
        #     arr+=rr
        #     air+=ir
        #     print( "conf_threshold", thr, "registration recall:", rr, " Inlier rate:", ir, "FMR:", fmr)

        # print("average registration recall:", arr / n, afmr/n, air/n)
        # print ("registration recall:", self.test_thr())

    def test_thr(self, iteration, conf_threshold=None):
        self.log_file.write(f"\n===============================\n     Iteration: {iteration}   conf_threshold: {conf_threshold}\n=============================\n")
        # print('Start to evaluate on test datasets...')
        # os.makedirs(f'{self.snapshot_dir}/{self.config.dataset}',exist_ok=True)

        num_iter = math.ceil(len(self.loader['test'].dataset) // self.loader['test'].batch_size)
        num_iter = 2
        c_loader_iter = self.loader['test'].__iter__()


        self.model.eval()




        success1 = 0.
        IR=0.
        FMR=0.

        with torch.no_grad():
            for idx in tqdm(range(num_iter)): # loop through this epoch

                ##################################
                if self.timers: self.timers.tic('load batch')
                try:
                    inputs = c_loader_iter.next()
                except:
                    return
                for k, v in inputs.items():
                    if type(v) == list:
                        inputs[k] = [item.to(self.device) for item in v]
                    elif type(v) in [dict, float, type(None), np.ndarray]:
                        pass
                    else:
                        inputs[k] = v.to(self.device)
                if self.timers: self.timers.toc('load batch')
                ##################################

                if self.timers: self.timers.tic('forward pass')
                data = self.model(inputs, timers=self.timers)  # [N1, C1], [N2, C2]
                if self.timers: self.timers.toc('forward pass')

                print("Number of source points: ", data['s_pcd'].cpu().numpy().shape)
                print("Number of target points: ", data['t_pcd'].cpu().numpy().shape)
                self.log_file.write(f"Number of source points: {data['s_pcd'].cpu().numpy().shape}\n")
                self.log_file.write(f"Number of target points: {data['t_pcd'].cpu().numpy().shape}\n")


                match_pred, masked_confidence_scores, mask = CM.get_match(data['conf_matrix_pred'], thr=conf_threshold, mutual=False)
                conf_tmp = masked_confidence_scores.cpu().numpy()
                print(f"Matching confidence:\n \t min = {np.min(conf_tmp)},\n \t max = {np.max(conf_tmp)},\n \t average = {np.mean(conf_tmp)},\n \t standard deviation = {np.std(conf_tmp)}")
                self.log_file.write(f"\n\nMatching confidence:\n \t min = {np.min(conf_tmp)},\n \t max = {np.max(conf_tmp)},\n \t average = {np.mean(conf_tmp)},\n \t standard deviation = {np.std(conf_tmp)}\n\n")
                mask_tmp = mask.cpu().numpy()
                if len(mask_tmp.shape) == 3:
                    assert mask_tmp.shape[0] == 1
                    mask_tmp = mask_tmp[0]
                plt.imshow(mask_tmp, cmap="seismic")
                plt.grid()
                plt.colorbar(label="Matching success (1) / failure (0)", orientation="horizontal")
                plt.title("3DMatch Registration Mask")
                plt.show()
                # import pdb; pdb.set_trace()
                rot, trn = MML.ransac_regist_coarse(data['s_pcd'], data['t_pcd'], data['src_mask'], data['tgt_mask'], match_pred)
                print(f"Rotation matrix: {rot}\n Translation vector: {trn}")
                self.log_file.write(f"Rotation matrix: {rot}\n Translation vector: {trn}\n")
                # ir = MML.compute_inlier_ratio(match_pred, data, inlier_thr=0.1).mean()

                # rr1 = MML.compute_registration_recall(rot, trn, data, thr=0.2) # 0.2m



                vis = True
                if vis:
                    pcd = data['points'][0].cpu().numpy()
                    lenth = data['stack_lengths'][0][0]
                    spcd, tpcd = pcd[:lenth] , pcd[lenth:]

                    import mayavi.mlab as mlab
                    c_red = (224. / 255., 0 / 255., 125 / 255.)
                    c_pink = (224. / 255., 75. / 255., 232. / 255.)
                    c_blue = (0. / 255., 0. / 255., 255. / 255.)
                    scale_factor = 0.02
                    # import pdb; pdb.set_trace()
                    # mlab.points3d(s_pc[ :, 0]  , s_pc[ :, 1],  s_pc[:,  2],  scale_factor=scale_factor , color=c_blue)
                    mlab.points3d(spcd[:, 0], spcd[:, 1], spcd[:, 2], scale_factor=scale_factor,
                                  color=c_red)
                    mlab.points3d(tpcd[:, 0], tpcd[:, 1], tpcd[:, 2], scale_factor=scale_factor,
                                  color=c_blue)
                    mlab.title("Original source and target pclouds.")
                    mlab.show()
                    reg_spcd = ( np.matmul(rot, spcd.T) + trn ).T
                    mlab.points3d(reg_spcd[:, 0], reg_spcd[:, 1], reg_spcd[:, 2], scale_factor=scale_factor,
                                  color=c_red)
                    mlab.points3d(tpcd[:, 0], tpcd[:, 1], tpcd[:, 2], scale_factor=scale_factor,
                                  color=c_blue)
                    mlab.title("Aligned source and target pclouds.")
                    mlab.show()

                    # if iteration == 0:
                    viz_pcd = o3d.geometry.PointCloud()
                    viz_pcd.points = o3d.utility.Vector3dVector(spcd)
                    viz_pcd.paint_uniform_color([255, 255, 0])
                    o3d.io.write_point_cloud(os.path.join(self.results_dir, f"source_pc_iteration-{iteration}_confthresh-{conf_threshold}_sample-{idx}.ply"), viz_pcd, write_ascii=True)

                    viz_pcd = o3d.geometry.PointCloud()
                    viz_pcd.points = o3d.utility.Vector3dVector(tpcd)
                    viz_pcd.paint_uniform_color([0, 255, 0])
                    o3d.io.write_point_cloud(os.path.join(self.results_dir, f"target_pc_iteration-{iteration}_confthresh-{conf_threshold}_sample-{idx}.ply"), viz_pcd, write_ascii=True)

                    viz_pcd = o3d.geometry.PointCloud()
                    viz_pcd.points = o3d.utility.Vector3dVector(reg_spcd)
                    viz_pcd.paint_uniform_color([0, 0, 255])
                    o3d.io.write_point_cloud(os.path.join(self.results_dir, f"reg_output_3dmatch_iteration-{iteration}_confthresh-{conf_threshold}_sample-{idx}.ply"), viz_pcd, write_ascii=True)
                    print(f"Registration output saved successfully in {self.results_dir}!!!")

                # return
            #     bs = len(rot)
            #     assert  bs==1
            #     success1 += bs * rr1
            #     IR += bs*ir
            #     FMR += (ir>0.05).float()


            # recall1 = success1/len(self.loader['test'].dataset)
            # IRate = IR/len(self.loader['test'].dataset)
            # FMR = FMR/len(self.loader['test'].dataset)
            # return recall1, IRate, FMR


def blend_anchor_motion (query_loc, reference_loc, reference_flow , knn=3, search_radius=0.1) :
    '''approximate flow on query points
    this function assume query points are sub- or un-sampled from reference locations
    @param query_loc:[m,3]
    @param reference_loc:[n,3]
    @param reference_flow:[n,3]
    @param knn:
    @return:
        blended_flow:[m,3]
    '''
    from datasets.utils import knn_point_np
    dists, idx = knn_point_np (knn, reference_loc, query_loc)
    dists[dists < 1e-10] = 1e-10
    mask = dists>search_radius
    dists[mask] = 1e+10
    weight = 1.0 / dists
    weight = weight / np.sum(weight, -1, keepdims=True)  # [B,N,3]
    blended_flow = np.sum (reference_flow [idx] * weight.reshape ([-1, knn, 1]), axis=1, keepdims=False)

    mask = mask.sum(axis=1)<3

    return blended_flow, mask

def compute_nrfmr( match_pred, data, recall_thr=0.04):


    s_pcd, t_pcd = data['s_pcd'], data['t_pcd']

    s_pcd_raw = data ['src_pcd_list']
    sflow_list = data['sflow_list']
    metric_index_list = data['metric_index_list']

    batched_rot = data['batched_rot']  # B,3,3
    batched_trn = data['batched_trn']


    nrfmr = 0.

    for i in range ( len(s_pcd_raw)):

        # get the metric points' transformed position
        metric_index = metric_index_list[i]
        sflow = sflow_list[i]
        s_pcd_raw_i = s_pcd_raw[i]
        metric_pcd = s_pcd_raw_i [ metric_index ]
        metric_sflow = sflow [ metric_index ]
        metric_pcd_deformed = metric_pcd + metric_sflow
        metric_pcd_wrapped_gt = ( torch.matmul( batched_rot[i], metric_pcd_deformed.T) + batched_trn[i] ).T


        # use the match prediction as the motion anchor
        match_pred_i = match_pred[ match_pred[:, 0] == i ]
        s_id , t_id = match_pred_i[:,1], match_pred_i[:,2]
        s_pcd_matched= s_pcd[i][s_id]
        t_pcd_matched= t_pcd[i][t_id]
        motion_pred = t_pcd_matched - s_pcd_matched
        metric_motion_pred, valid_mask = blend_anchor_motion(
            metric_pcd.cpu().numpy(), s_pcd_matched.cpu().numpy(), motion_pred.cpu().numpy(), knn=3, search_radius=0.1)
        metric_pcd_wrapped_pred = metric_pcd + torch.from_numpy(metric_motion_pred).to(metric_pcd)

        debug = False
        if debug:
            import mayavi.mlab as mlab
            c_red = (224. / 255., 0 / 255., 125 / 255.)
            c_pink = (224. / 255., 75. / 255., 232. / 255.)
            c_blue = (0. / 255., 0. / 255., 255. / 255.)
            scale_factor = 0.013
            metric_pcd_wrapped_gt = metric_pcd_wrapped_gt.cpu()
            metric_pcd_wrapped_pred = metric_pcd_wrapped_pred.cpu()
            err = metric_pcd_wrapped_pred - metric_pcd_wrapped_gt
            mlab.points3d(metric_pcd_wrapped_gt[:, 0], metric_pcd_wrapped_gt[:, 1], metric_pcd_wrapped_gt[:, 2], scale_factor=scale_factor, color=c_pink)
            mlab.points3d(metric_pcd_wrapped_pred[ :, 0] , metric_pcd_wrapped_pred[ :, 1], metric_pcd_wrapped_pred[:,  2], scale_factor=scale_factor , color=c_blue)
            mlab.quiver3d(metric_pcd_wrapped_gt[:, 0], metric_pcd_wrapped_gt[:, 1], metric_pcd_wrapped_gt[:, 2], err[:, 0], err[:, 1], err[:, 2],
                          scale_factor=1, mode='2ddash', line_width=1.)
            mlab.show()

        dist = torch.sqrt( torch.sum( (metric_pcd_wrapped_pred - metric_pcd_wrapped_gt)**2, dim=1 ) )

        r = (dist < recall_thr).float().sum() / len(dist)
        nrfmr = nrfmr + r

    nrfmr = nrfmr /len(s_pcd_raw)

    return  nrfmr

class _4DMatchTester(Trainer):
    """
    4DMatch tester
    """
    def __init__(self,args):
        Trainer.__init__(self, args)
        expt_dir = time.strftime("%D_%H-%M", time.localtime(time.time())).replace('/','-')
        self.results_dir = os.path.join(self.config.results_dir, expt_dir)
        os.makedirs(self.results_dir)
        self.log_file = open(os.path.join(self.results_dir, "4d_match_results.txt"), "w+")

    def test(self):

        for i, thr in enumerate([0.1]):
        # for i, thr in enumerate([0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]):
        # for thr in [ 0.1 ]:
            import time
            start = time.time()
            self.test_thr(i, thr)
            # ir, fmr, nspl = self.test_thr(thr)
            # print( "conf_threshold", thr,  "NFMR:", fmr, " Inlier rate:", ir, "Number sample:", nspl)
            # print( "time costs:", time.time() - start)
        self.log_file.close()

    def test_thr(self, iteration, conf_threshold=None):
        self.log_file.write(f"\n===============================\n     Iteration: {iteration}   conf_threshold: {conf_threshold}\n=============================\n")

        num_iter = math.ceil(len(self.loader['test'].dataset) // self.loader['test'].batch_size)
        print("Number of iterations: ", num_iter)
        # import pdb;pdb.set_trace()
        num_iter = 1
        c_loader_iter = self.loader['test'].__iter__()


        self.model.eval()


        assert self.loader['test'].batch_size == 1

        IR=0.
        NR_FMR=0.

        inlier_thr = recall_thr = 0.04

        n_sample = 0.

        with torch.no_grad():
            for idx in tqdm(range(num_iter)): # loop through this epoch



                ##################################
                if self.timers: self.timers.tic('load batch')
                try:
                    inputs = c_loader_iter.next()
                except:
                    return
                #import pdb; pdb.set_trace()
                for k, v in inputs.items():
                    if type(v) == list:
                        inputs[k] = [item.to(self.device) for item in v]
                    elif type(v) in [ dict, float, type(None), np.ndarray]:
                        pass
                    else:
                        inputs[k] = v.to(self.device)
                if self.timers: self.timers.toc('load batch')
                # import pdb; pdb.set_trace()
                ##################################


                #import pdb; pdb.set_trace()

                if self.timers: self.timers.tic('forward pass')
                data = self.model(inputs, timers=self.timers)  # [N1, C1], [N2, C2]



                #import pdb; pdb.set_trace()
                print("Number of source points: ", data['s_pcd'].cpu().numpy().shape)
                print("Number of target points: ", data['t_pcd'].cpu().numpy().shape)
                self.log_file.write(f"Number of source points: {data['s_pcd'].cpu().numpy().shape}\n")
                self.log_file.write(f"Number of target points: {data['t_pcd'].cpu().numpy().shape}\n")


                if self.timers: self.timers.toc('forward pass')
                print("Confidence threshold: ", conf_threshold)
                self.log_file.write(f"\n\nConfidence threshold: {conf_threshold}\n")
                match_pred, masked_confidence_scores, mask = CM.get_match(data['conf_matrix_pred'], thr=conf_threshold, mutual=True)


                conf_tmp = masked_confidence_scores.cpu().numpy()
                print(f"Matching confidence:\n \t min = {np.min(conf_tmp)},\n \t max = {np.max(conf_tmp)},\n \t average = {np.mean(conf_tmp)},\n \t standard deviation = {np.std(conf_tmp)}")
                self.log_file.write(f"\n\nMatching confidence:\n \t min = {np.min(conf_tmp)},\n \t max = {np.max(conf_tmp)},\n \t average = {np.mean(conf_tmp)},\n \t standard deviation = {np.std(conf_tmp)}\n\n")
                mask_tmp = mask.cpu().numpy()
                if len(mask_tmp.shape) == 3:
                    assert mask_tmp.shape[0] == 1
                    mask_tmp = mask_tmp[0]
                plt.imshow(mask_tmp, cmap="seismic")
                plt.grid()
                plt.colorbar(label="Matching success (1) / failure (0)", orientation="horizontal")
                plt.title("4DMatch Registration Mask")
                plt.show()

                rot, trn = data["R_s2t_pred"].cpu().numpy(), data["t_s2t_pred"].cpu().numpy()
                print(f"Rotation matrix: {rot}")
                self.log_file.write(f"Rotation matrix: {rot}")
                print("Translation matrix: ", trn)
                self.log_file.write(f"Translation matrix: {trn}")

                vis = True
                if vis:
                    pcd = data['points'][0].cpu().numpy()
                    lenth = data['stack_lengths'][0][0]
                    spcd, tpcd = pcd[:lenth] , pcd[lenth:]

                    import mayavi.mlab as mlab
                    c_red = (224. / 255., 0 / 255., 125 / 255.)
                    c_pink = (224. / 255., 75. / 255., 232. / 255.)
                    c_blue = (0. / 255., 0. / 255., 255. / 255.)
                    scale_factor = 0.02
                    # mlab.points3d(s_pc[ :, 0]  , s_pc[ :, 1],  s_pc[:,  2],  scale_factor=scale_factor , color=c_blue)
                    mlab.points3d(spcd[:, 0], spcd[:, 1], spcd[:, 2], scale_factor=scale_factor,
                                  color=c_red)
                    mlab.points3d(tpcd[:, 0], tpcd[:, 1], tpcd[:, 2], scale_factor=scale_factor,
                                  color=c_blue)
                    mlab.title("Original source and target pclouds.")
                    mlab.show()

                    # import pdb; pdb.set_trace()
                    reg_spcd = ( np.matmul(rot, spcd.T) + trn ).T



                    mlab.points3d(reg_spcd[:, 0], reg_spcd[:, 1], reg_spcd[:, 2], scale_factor=scale_factor,
                                  color=c_red)
                    mlab.points3d(tpcd[:, 0], tpcd[:, 1], tpcd[:, 2], scale_factor=scale_factor,
                                  color=c_blue)
                    mlab.title("Aligned source and target pclouds.")
                    mlab.show()

                    # if iter == 0:
                    viz_pcd = o3d.geometry.PointCloud()
                    viz_pcd.points = o3d.utility.Vector3dVector(spcd)
                    viz_pcd.paint_uniform_color([255, 255, 0])
                    o3d.io.write_point_cloud(os.path.join(self.results_dir, f"source_pc_iteration-{iteration}_confthresh-{conf_threshold}_sample-{idx}.ply"), viz_pcd, write_ascii=True)

                    viz_pcd = o3d.geometry.PointCloud()
                    viz_pcd.points = o3d.utility.Vector3dVector(tpcd)
                    viz_pcd.paint_uniform_color([0, 255, 0])
                    o3d.io.write_point_cloud(os.path.join(self.results_dir, f"target_pc_iteration-{iteration}_confthresh-{conf_threshold}_sample-{idx}.ply"), viz_pcd, write_ascii=True)

                    viz_pcd = o3d.geometry.PointCloud()
                    viz_pcd.points = o3d.utility.Vector3dVector(reg_spcd)
                    viz_pcd.paint_uniform_color([0, 0, 255])
                    o3d.io.write_point_cloud(os.path.join(self.results_dir, f"reg_output_4dmatch_iteration-{iteration}_confthresh-{conf_threshold}_sample-{idx}.ply"), viz_pcd, write_ascii=True)
                    print(f"Registration output saved successfully in {self.results_dir}!!!")

            #     ir = MML.compute_inlier_ratio(match_pred, data, inlier_thr=inlier_thr, s2t_flow=data['coarse_flow'][0][None] )[0]

            #     nrfmr = compute_nrfmr(match_pred, data, recall_thr=recall_thr)

            #     IR += ir
            #     NR_FMR += nrfmr

            #     n_sample += match_pred.shape[0]


            # IRate = IR/len(self.loader['test'].dataset)
            # NR_FMR = NR_FMR/len(self.loader['test'].dataset)
            # n_sample = n_sample/len(self.loader['test'].dataset)

            # if self.timers: self.timers.print()

            # return IRate, NR_FMR, n_sample





def get_trainer(config):
    if config.dataset == '3dmatch':
        return _3DMatchTester(config)
    elif config.dataset == '4dmatch':
        return _4DMatchTester(config)
    else:
        raise NotImplementedError
