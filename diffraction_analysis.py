import cv2
from tqdm import tqdm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.patches import Circle


#from utilities import create_circular_mask

from serving_manager.management.torchserve_rest_manager import TorchserveRestManager


def smart_VDF(image,image_array,threshold=0,host="192.168.51.3"):

    try:
        manager = TorchserveRestManager(inference_port='8080', management_port='8081', host=host,
                                        image_encoder='.tiff') #start server manager
        manager.scale(model_name="spot_segmentation") #scale the server to have 2 processes
        results = manager.infer(image=image, model_name='spot_segmentation')  # send image to spot detection
        spots = results["objects"] #retrieve spot details
    except ConnectionError:
        print("Could not connect to model, check the host and if there are available workers")
    spot_list = [] #list for all spots
    areas = []
    spot_radii = []

    shape = image.shape
    for i in range(len(spots)):
        if spots[i]["mean_intensity"] > threshold:
            spot_coords = spots[i]["center"] #spot center in fractions of image
            spot_list.append(spot_coords)
            area = spots[i]["area"] #spot area in fractional dimensions
            areas.append(area)
            radius = np.sqrt(area/np.pi)/shape[0] #calculate radius from area
            spot_radii.append(radius)



        if type(image_array) is tuple:  # checks for metadata dictionary
            image_array = image_array[0]

        else:
            image_array = image_array


        dataset_shape = image_array.shape[0], image_array.shape[1]
        dp_shape = image_array[0][0].shape

        all_mask_intensities = []  # empty list for the mask data

        integration_masks = []
        for mask in range(len(spot_list)):
            mask_coords = spot_list[mask][0]*512,spot_list[mask][1]*512
            print(f"spot position {spot_list[mask]} mask coords{mask_coords}")
            radius = spot_radii[mask]*512
            print(f"measured radius {spot_radii[mask]} radius {radius}")
            integration_mask = create_circular_mask(dp_shape[0], dp_shape[1], mask_center_coordinates=mask_coords,
                                                    mask_radius=radius)
            integration_masks.append(integration_mask)

        for row in tqdm(image_array):
            for pixel in row:
                mask_intensities = []  # empty list of mask intensities per pixel
                for mask_coords in range(len(spot_list)):  # for each mask
                    integration_mask = integration_masks[mask_coords]  # takes mask from list and applies it
                    mask_intensity = np.sum(pixel[integration_mask])  # measures the intensity in the masked image
                    mask_intensities.append(mask_intensity)  # adds to the list

                all_mask_intensities.append(mask_intensities)  # adds each pixels list to a list (nested lists)

        DF_images = []  # empty list for DF images to be added to
        for mask in range(len(spot_list)):  # for every mask in the list
            DF_output = [i[mask] for i in all_mask_intensities]  # this works but I don't really understand why
            DF_output = np.reshape(DF_output, (dataset_shape))  # reshapes the DF intensities to the scan dimensions
            DF_images.append(DF_output)  # adds DF images to a list

        #fig,ax = plt.subplots(1,1)
    #for i in range(len(spot_list)):
    #    ax.plot(spot_list[i][0],spot_list[i][1],"r+") #plot spot center with red cross marker
    #    spot_marker = Circle(xy=(spot_list[i][0],spot_list[i][1]),radius=spot_radii[i],color="yellow",fill=False)
    #    ax.add_patch(spot_marker)
    #    plt.imshow(image,vmax=np.average(image*10),extent=(0,1,1,0),cmap="gray")
    output= (spot_list,spot_radii,DF_images)

    return output




# ------------------ Core lattice algebra ------------------

def gauss_reduce_2d(A):
    """Gauss reduction of a 2D basis matrix (2x2)."""
    a = A[:,0].copy(); b = A[:,1].copy()
    if np.linalg.norm(a) > np.linalg.norm(b): a,b = b,a
    while True:
        mu = np.round(np.dot(a,b)/np.dot(a,a))
        b = b - mu*a
        if np.linalg.norm(b) < np.linalg.norm(a): a,b = b,a; continue
        if abs(np.dot(a,b)) <= 0.5*np.dot(a,a): break
    return np.column_stack((a,b))

def orient_basis(A, quadrant=(+1,+1)):
    """Flip basis vectors to prefer pointing into the given quadrant."""
    B = A.copy(); wantx,wanty = quadrant
    for i in (0,1):
        v = B[:,i]
        if (v[0] == 0 and wanty*v[1] < 0) or (v[0]*wantx < 0):
            B[:,i] = -v
    return B

# ------------------ Helpers ------------------

def median_nn_spacing(points):
    pts = np.asarray(points, float)
    tree = cKDTree(pts); d,_ = tree.query(pts, k=2)
    return float(np.median(d[:,1]))

def neighbor_differences(points, r_max):
    pts = np.asarray(points, float)
    tree = cKDTree(pts)
    diffs = []
    for i, p in enumerate(pts):
        for j in tree.query_ball_point(p, r_max):
            if j > i:
                diffs.append(pts[j]-p)
    return np.array(diffs) if len(diffs) else np.zeros((0,2))

# ------------------ Clustering differences ------------------

def cluster_differences(points, r_factor=2.2, eps_frac=0.12, min_samples=15):
    """
    Cluster neighbor differences using DBSCAN.
    Returns: (diffs, db, centers, nn_spacing)
    """
    pts = np.asarray(points, float)
    nn = median_nn_spacing(pts)
    diffs = neighbor_differences(pts, r_max=r_factor*nn)
    if len(diffs)==0:
        raise RuntimeError("No neighbor differences; increase r_factor.")
    V = np.vstack([diffs, -diffs])   # include ±
    eps = eps_frac * nn
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(V)
    labels = db.labels_
    centers = []
    for lbl in set(labels):
        if lbl == -1: continue
        S = V[labels==lbl]
        center = np.median(S, axis=0)
        Lmed   = float(np.median(np.linalg.norm(S, axis=1)))
        centers.append((center, Lmed, len(S)))
    centers.sort(key=lambda t: t[1])  # shortest first
    return diffs, db, centers, nn

# ------------------ Classification ------------------

def classify_with_offset(points, A, t, tol_frac=0.07, idx_margin=2):
    """Classify points as periodic vs outlier with basis A and offset t."""
    pts = np.asarray(points, float)
    Ainv = np.linalg.inv(A)
    n = (Ainv @ (pts - t).T).T
    N = np.rint(n)
    recon = (A @ N.T).T + t
    resid = np.linalg.norm(pts - recon, axis=1)
    spacing = min(np.linalg.norm(A[:,0]), np.linalg.norm(A[:,1]))
    tol = tol_frac * max(spacing, 1e-12)
    mask = resid <= tol
    if np.any(mask):
        Ni = N[mask].astype(int)
        mn, mx = Ni.min(axis=0), Ni.max(axis=0)
        inside = np.all((N >= mn-idx_margin) & (N <= mx+idx_margin), axis=1)
        mask = mask & inside
    return mask, resid, tol, N

# ------------------ Refinement ------------------

def refine_basis_offset(points, A_init, t_init=None,
                        iters=12, tol_schedule=(0.5,0.35,0.25,0.2,0.15)):
    """
    Iteratively refine both basis A and translation offset t.
    Returns (A_refined, t_refined).
    """
    pts = np.asarray(points, float)
    A = A_init.copy()
    t = np.zeros(2) if t_init is None else t_init.copy()
    for k in range(iters):
        tol_frac = tol_schedule[min(k, len(tol_schedule)-1)]
        Ainv = np.linalg.inv(A)
        n = (Ainv @ (pts - t).T).T
        N = np.rint(n)
        recon = (A @ N.T).T + t
        resid = np.linalg.norm(pts - recon, axis=1)
        spacing = min(np.linalg.norm(A[:,0]), np.linalg.norm(A[:,1]))
        tol = tol_frac * max(spacing, 1e-12)
        mask = resid < tol
        if mask.sum() < 5:
            continue
        X = np.hstack([N[mask], np.ones((mask.sum(),1))])
        Y = pts[mask]
        M2 = X.T @ X
        if np.linalg.cond(M2) > 1e10: break
        M1 = Y.T @ X
        At = M1 @ np.linalg.inv(M2)
        A_new = At[:, :2]; t_new = At[:, 2]
        A = orient_basis(gauss_reduce_2d(0.6*A + 0.4*A_new))
        t = 0.6*t + 0.4*t_new
    return orient_basis(gauss_reduce_2d(A)), t

# ------------------ Candidate search ------------------

def score_basis(points, A, t, tol_frac=0.08, nn=None):
    """
    Score a basis by number of inliers, residuals, and primitive penalty.
    Returns (score, inliers, mask, resid).
    """
    mask, resid, _, _ = classify_with_offset(points, A, t, tol_frac=tol_frac)
    inliers = int(mask.sum())
    if inliers == 0:
        return -np.inf, inliers, mask, resid

    # residual quality
    res_score = -np.median(resid[mask])

    # primitive penalty: prefer short vectors near NN spacing
    if nn is None:
        nn = median_nn_spacing(points)
    lengths = [np.linalg.norm(A[:,0]), np.linalg.norm(A[:,1])]
    penalties = []
    for L in lengths:
        if L < 0.5*nn:            # suspiciously short (noise cluster)
            penalties.append(-1.0)
        elif L > 2.0*nn:          # likely harmonic multiple
            penalties.append(-0.5 * (L/nn))
        else:
            penalties.append(0.0)
    prim_penalty = sum(penalties)

    # total score combines all
    score = res_score + 0.01*inliers + prim_penalty
    return score, inliers, mask, resid


def best_basis_from_candidates(points, centers, nn, max_pairs=200):
    """
    Search for the best primitive lattice basis from clustered differences.
    Raises RuntimeError only if there is genuinely no usable non-collinear pair.
    """

    pts = np.asarray(points, float)

    # ---- collapse ± duplicates ----
    def canonical(v):
        v = v / np.linalg.norm(v)
        if v[0] < 0 or (v[0] == 0 and v[1] < 0):
            v = -v
        return tuple(np.round(v, 6))

    seen = set(); uniq = []
    for center, L, n in centers:
        key = canonical(center)
        if key not in seen:
            seen.add(key)
            uniq.append((center, L, n))
    centers = uniq

    # ---- build candidate pairs ----
    pairs = []
    for i in range(len(centers)):
        for j in range(i+1, len(centers)):
            v1, v2 = centers[i][0], centers[j][0]
            cosang = abs(np.dot(v1, v2) /
                         (np.linalg.norm(v1)*np.linalg.norm(v2)))
            if cosang > 0.9995:  # only reject almost exact parallel
                continue
            pairs.append((v1, v2))

    # ---- fallback if no pairs ----
    if not pairs:
        if len(centers) >= 2:
            v1, v2 = centers[0][0], centers[1][0]
            pairs = [(v1, v2)]
        else:
            raise RuntimeError(
                "No non-collinear candidate pairs (data may not be periodic)."
            )

    # ---- limit how many pairs we test ----
    rng = np.random.default_rng(0)
    rng.shuffle(pairs)
    pairs = pairs[:max_pairs]

    # ---- search best basis ----
    best = {"A": None, "t": None, "score": -np.inf,
            "inliers": 0, "mask": None, "resid": None}

    for v1, v2 in pairs:
        A0 = orient_basis(gauss_reduce_2d(np.column_stack((v1, v2))))
        # sanity: scale up too-short candidates
        for i in (0, 1):
            L = np.linalg.norm(A0[:, i])
            if L < 0.6 * nn:
                k = int(round(nn / max(L, 1e-12)))
                k = max(1, min(k, 3))
                A0[:, i] *= k

        Aref, tref = refine_basis_offset(pts, A0, iters=12)
        s, n, m, r = score_basis(pts, Aref, tref, tol_frac=0.10, nn=nn)


        if s > best["score"]:
            best.update(A=Aref, t=tref, score=s,
                        inliers=n, mask=m, resid=r)

    return best

def vector_match_quality(points, v, t=None):
    """
    Measure how well a single vector v explains periodicity of points.
    Returns mean residual (lower is better).
    """
    pts = np.asarray(points, float)
    if t is None:
        t = np.zeros(2)
    v = np.asarray(v, float)
    L = np.linalg.norm(v)
    if L < 1e-12:
        return np.inf

    proj = (pts - t) @ v / L  # projection along v
    resid = np.abs(proj - np.round(proj))
    return resid.mean()


def detect(points, r_factor=2.2, eps_frac=0.12, min_samples=12, debug=False):
    """
    Detect lattice in 2D point set.
    Returns dict with:
        A           -> 2x2 primitive basis (best fit)
        t           -> translation offset
        mask        -> boolean array (periodic vs outlier)
        resid       -> residuals
        tol         -> tolerance used for classification
        directions  -> list of representative directions (shortest per group)
        groups      -> merged direction groups with harmonics
        scores      -> list of (pair_idx, score, inliers) for candidate bases
    """
    pts = np.asarray(points, float)

    # --- Step 1: Cluster differences ---
    diffs, db, centers, nn = cluster_differences(pts,
                                                 r_factor=r_factor,
                                                 eps_frac=eps_frac,
                                                 min_samples=min_samples)

    if len(centers) < 2:
        # retry with looser settings
        diffs, db, centers, nn = cluster_differences(pts,
                                                     r_factor=max(r_factor, 2.0),
                                                     eps_frac=eps_frac*0.7,
                                                     min_samples=max(5, min_samples//2))
        if len(centers) < 2:
            raise RuntimeError("Not enough distinct directions to form a lattice basis.")

    # --- Step 2: Merge collinear directions ---
    groups = merge_collinear_directions(centers, angle_deg=3.0, len_tol=0.12)

    # representatives (shortest in each group)
    uniq = [(g["rep"], np.linalg.norm(g["rep"]), len(g["members"])) for g in groups]

    if debug:
        print(f"[DEBUG] groups found: {len(groups)}")
        for gi,g in enumerate(groups):
            repL = np.linalg.norm(g["rep"])
            print(f"  Group {gi}: rep length≈{repL:.3f}, members={len(g['members'])}, harmonics={len(g['harmonics'])}")

    if len(uniq) < 2:
        raise RuntimeError("Still not enough usable directions after merging.")

    # --- Step 3/4: Test all non-collinear representative pairs ---
    pairs = []
    for i in range(len(uniq)):
        for j in range(i+1, len(uniq)):
            v1, v2 = uniq[i][0], uniq[j][0]
            cosang = abs(np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2)))
            if cosang < 0.995:  # non-collinear
                pairs.append((i, j, v1, v2))

    if not pairs:
        raise RuntimeError("No non-collinear representative pairs (data may not be periodic).")

    best = {"A": None, "t": None, "score": -np.inf, "inliers": 0, "mask": None, "resid": None}
    scores = []

    for (i, j, v1, v2) in pairs:
        A0 = gauss_reduce_2d(np.column_stack((v1, v2)))
        Aref, tref = refine_basis_offset(pts, A0, iters=12)
        s, n, m, r = score_basis(pts, Aref, tref, tol_frac=0.10)

        scores.append(((i, j), s, n))

        if s > best["score"]:
            best.update(A=Aref, t=tref, score=s, inliers=n, mask=m, resid=r)

    if best["A"] is None:
        raise RuntimeError("Failed to refine any candidate basis.")

    # --- Step 5: Classification tolerance ---
    tol = 0.08 * min(np.linalg.norm(best["A"][:,0]), np.linalg.norm(best["A"][:,1]))

    # --- Step 6: Return ---
    return {
        "A": best["A"],
        "t": best["t"],
        "mask": best["mask"],
        "resid": best["resid"],
        "tol": tol,
        "directions": [c for c,L,n in uniq],  # representatives
        "groups": groups,
        "scores": scores
    }


def merge_collinear_directions(centers, angle_deg=3.0, len_tol=0.12):
    """
    Merge near-collinear direction clusters and detect harmonics.
    centers: list of (center_vec, L_med, count)

    Returns:
      groups: list of dicts:
        {
          "rep": np.array(2,),     # representative direction (shortest in group)
          "members": [(v,L,n),...],
          "harmonics": [ (k, v, L) ]  # k ~ integer multiple of rep length
        }
    """

    ang_tol = math.radians(angle_deg)

    # compute angles mod pi (direction up to sign)
    def angle_mod_pi(v):
        th = math.atan2(v[1], v[0])
        if th < 0: th += math.pi
        return th

    # normalize sign so x>=0 (for consistent angle)
    def normalize_sign(v):
        if v[0] < 0 or (v[0] == 0 and v[1] < 0):
            return -v
        return v

    items = []
    for v,L,n in centers:
        vv = normalize_sign(np.asarray(v, float))
        items.append((vv, float(L), int(n), angle_mod_pi(vv)))

    # sort by angle then by length
    items.sort(key=lambda t: (t[3], t[1]))

    groups = []
    for v,L,n,theta in items:
        placed = False
        for g in groups:
            vg = g["rep"]
            # angle between v and group rep
            cosang = abs(np.dot(v, vg)/(np.linalg.norm(v)*np.linalg.norm(vg)))
            if cosang > np.cos(ang_tol):
                g["members"].append((v,L,n))
                # update rep if this member is shorter
                if L < np.linalg.norm(g["rep"]):
                    g["rep"] = v
                placed = True
                break
        if not placed:
            groups.append({"rep": v.copy(), "members": [(v,L,n)], "harmonics": []})

    # within each group, mark harmonics relative to the shortest rep
    for g in groups:
        L0 = np.linalg.norm(g["rep"])
        for v,L,n in g["members"]:
            k = round(L / L0) if L0 > 1e-12 else 1
            if k >= 2 and abs(L - k*L0) <= len_tol * L0:
                g["harmonics"].append((k, v, L))

    return groups




def find_origin(points, target=(0.5, 0.5)):
    pts = np.asarray(points)
    target = np.array(target)
    dists = np.linalg.norm(pts - target, axis=1)
    return pts[np.argmin(dists)]


def plot_result(points, result,image=None, title="Lattice detection result", zoom_factor=10):
    """
    Plot lattice detection results.
    - True-length arrows on main plot
    - Length + quality labels on arrows
    - Inset zoom showing magnified arrows
    """
    pts = np.asarray(points)
    mask = result["mask"]
    A = result["A"]
    groups = result.get("groups", [])
    tol = result["tol"]
    resid = result["resid"]
    origin = find_origin(points, target=(0.5, 0.5))

    fig, axs = plt.subplots(1,2, figsize=(12,6))

    # =========== MAIN PLOT ===========
    ax = axs[0]
    ax.set_title(title)
    ax.scatter(pts[~mask,0], pts[~mask,1], c="red", s=25, label="Outliers")
    ax.scatter(pts[mask,0], pts[mask,1], c="green", s=12, label="Periodic")

    if image is not None:
        ax.imshow(image,vmax=(np.average(image)*0.2),extent=(0,1,1,0))

    # representatives and harmonics
    for gi,g in enumerate(groups):
        rep = g["rep"]
        rep_q = vector_match_quality(points, rep, result["t"])
        rep_len = np.linalg.norm(rep)

        # representative arrow
        ax.arrow(origin[0], origin[1], rep[0], rep[1],
                 head_width=0.002, head_length=0.003,
                 fc="gray", ec="gray", lw=1.5, alpha=0.8)
        ax.text(origin[0]+rep[0]*1.1, origin[1]+rep[1]*1.1,
                f"G{gi}\nL={rep_len:.3f}\nq={rep_q:.2f}",
                color="black", fontsize=7, ha="center")

        # harmonics
        for (k, v, L) in g["harmonics"]:
            q = vector_match_quality(points, v, result["t"])
            ax.arrow(origin[0], origin[1], v[0], v[1],
                     head_width=0.0015, head_length=0.002,
                     fc="none", ec="gray", lw=1, alpha=0.4,
                     linestyle="--")
            ax.text(origin[0]+v[0]*1.1, origin[1]+v[1]*1.1,
                    f"k={k}\nL={L:.3f}\nq={q:.2f}",
                    color="dimgray", fontsize=6, ha="center")

    # highlight primitive basis
    for vec,col in zip(A.T, ["tab:orange","tab:blue"]):
        ax.arrow(origin[0], origin[1], vec[0], vec[1],
                 head_width=0.003, head_length=0.004, fc=col, ec=col, lw=2)

    ax.legend()
    ax.set_aspect("equal","box")

    # =========== INSET ZOOM ===========
    ax_inset = inset_axes(ax, width="30%", height="30%", loc="upper right")
    ax_inset.set_title("Vectors zoom", fontsize=8)
    ax_inset.set_xlim(-1,1)
    ax_inset.set_ylim(-1,1)
    ax_inset.set_aspect("equal","box")

    for gi,g in enumerate(groups):
        rep = g["rep"] * zoom_factor
        ax_inset.arrow(0, 0, rep[0], rep[1],
                       head_width=0.05, head_length=0.07,
                       fc="gray", ec="gray", lw=1.2, alpha=0.8)
        for (k, v, L) in g["harmonics"]:
            v_scaled = v * zoom_factor
            ax_inset.arrow(0, 0, v_scaled[0], v_scaled[1],
                           head_width=0.03, head_length=0.05,
                           fc="none", ec="gray", lw=1, alpha=0.4,
                           linestyle="--")

    for vec,col in zip(A.T, ["tab:orange","tab:blue"]):
        v = vec * zoom_factor
        ax_inset.arrow(0, 0, v[0], v[1],
                       head_width=0.06, head_length=0.08, fc=col, ec=col, lw=2)

    ax_inset.set_xticks([]); ax_inset.set_yticks([])

    # =========== RESIDUAL HISTOGRAM ===========
    axs[1].set_title("Residuals")
    axs[1].hist(resid, bins=40, color="gray", alpha=0.8)
    axs[1].axvline(tol, color="red", linestyle="--", label=f"tol={tol:.3g}")
    axs[1].legend()
    axs[1].set_xlabel("Residual distance")

    plt.tight_layout()
    plt.show()

def model_has_workers(model_name,host="192.168.51.3"):
    manager = TorchserveRestManager(inference_port='8080', management_port='8081', host=host,
                                    image_encoder='.tiff')  # contacts the model manager
    model_status = manager.describe_model(model_name)
    if model_status[0]["minWorkers"] != 0:
        has_workers = True
    else: has_workers = False

    return has_workers

from time import time

def get_spot_positions(image,threshold=0,host="192.168.51.3",model_name="spot_segmentation"):

    try:
        manager = TorchserveRestManager(inference_port='8080', management_port='8081', host=host,
                                        image_encoder='.tiff')  # start server manager
        if not model_has_workers(model_name,host=host):
            manager.scale(model_name=model_name) #scale the server to have 1 process

        process_pre = time()
        results = manager.infer(image=image, model_name=model_name)  # send image to spot detection
        inference_time = time()-process_pre
        print(f"Inference Time: {inference_time:3f}s")
        spots = results["objects"] #retrieve spot details
    except ConnectionError:
        print("Could not connect to model, check the model is available")
        return [],[]
    spot_list = [] #list for all spots
    areas = []
    spot_radii = []
    fig,ax = plt.subplots(1,1)
    shape = image.shape
    for spot in spots:
        if spot["mean_intensity"] > threshold:
            spot_coords = spot["center"]
            area = spot["area"]

            spot_list.append(spot_coords)
            areas.append(area)
            radius = np.sqrt(area / np.pi) / shape[0]
            spot_radii.append(radius)

    ax.imshow(image, vmax=np.average(image * 10), extent=(0, 1, 1, 0), cmap="gray")

    for coords, radius in zip(spot_list, spot_radii):
        ax.plot(coords[0], coords[1], "r+")
        spot_marker = Circle(xy=coords, radius=radius, color="yellow", fill=False)
        ax.add_patch(spot_marker)


    plt.show(block=False)
    output= (spot_list,spot_radii)

    return output

def run(host="192.168.51.3"):

    image = cv2.imread(r"C:\Users\robert.hooley\Documents\Coding\Datasets for development\filename26.tiff",cv2.IMREAD_UNCHANGED)

    results = get_spot_positions(image,host=host)

    detection = detect(results[0],debug=True)

    plot_result(results[0], detection,image=image, title="Basis vector detection")



def annular_angle_map(
    data4d: np.ndarray,
    r_inner: float,
    r_outer: float,
    *,
    center: tuple[float, float] | None = None,
    angle_bins: int = 180,
    log_intensity: bool = False,
    vmin_percentile: float = 1.0,
    vmax_percentile: float = 99.0,
    min_value=1.0,
    print_progress: bool = True,
    # debug / plotting controls
    dp_index: tuple[int, int] | None = None,   # (sy,sx) to show a DP with mask
    log_dp: bool = True,
    mask_alpha: float = 0.35,
    show_plots: bool = True,
    title: str | None = None,
    wheel_side: str = "right",                 # "right" or "left"
    wheel_px: int = 140,
    wheel_margin: float = 0.02,
) -> tuple[np.ndarray, dict, tuple[plt.Figure, plt.Axes, plt.Axes] | None, tuple[plt.Figure, plt.Axes] | None]:
    """
    One-stop function:
      1) Computes RGB scan image from 4D-STEM (Sy,Sx,Ky,Kx)
         - brightness = annulus total intensity
         - hue = angle bin with maximum intensity within the annulus
      2) Optionally plots:
         - RGB image with an external colorwheel legend
         - One diffraction pattern with annulus mask overlay

    Returns
    -------
    rgb : (Sy,Sx,3) float32 in [0,1]
    debug : dict (includes annulus_mask, annulus_total, argmax_bin, center, etc.)
    fig_rgb_tuple : (fig, ax_img, ax_wheel) or None
    fig_dp_tuple  : (fig, ax_dp) or None
    """
    if data4d.ndim != 4:
        raise ValueError(f"data4d must be 4D (Sy,Sx,Ky,Kx). Got shape {data4d.shape}")

    Sy, Sx, Ky, Kx = data4d.shape
    if center is None:
        cy = (Ky - 1) / 2.0
        cx = (Kx - 1) / 2.0
    else:
        cy, cx = center

    if print_progress:
        print(f"[annular] data shape: (Sy,Sx,Ky,Kx)=({Sy},{Sx},{Ky},{Kx})")
        print(f"[annular] center=(cy,cx)=({cy:.3f},{cx:.3f}), r_inner={r_inner}, r_outer={r_outer}, bins={angle_bins}")

    # Detector polar coordinates
    yy, xx = np.indices((Ky, Kx), dtype=np.float32)
    dy = yy - cy
    dx = xx - cx
    rr = np.sqrt(dy * dy + dx * dx)
    theta = np.arctan2(dy, dx)  # [-pi, pi]

    annulus_mask = (rr >= r_inner) & (rr <= r_outer)
    if not np.any(annulus_mask):
        raise ValueError("Annulus mask is empty. Check r_inner/r_outer/center.")

    M = int(np.count_nonzero(annulus_mask))
    if print_progress:
        print(f"[annular] annulus pixels: {M} / {Ky*Kx} ({100*M/(Ky*Kx):.2f}%)")

    # Flatten annulus pixels
    theta_a = theta[annulus_mask].astype(np.float32)  # (M,)
    # Angle -> bins
    u = (theta_a + np.pi) / (2.0 * np.pi)  # [0,1)
    bin_idx = np.floor(u * angle_bins).astype(np.int32)
    bin_idx = np.clip(bin_idx, 0, angle_bins - 1)

    if print_progress:
        print("[annular] building binning matrix...")
    bin_mat = np.zeros((M, angle_bins), dtype=np.float32)
    bin_mat[np.arange(M), bin_idx] = 1.0

    if print_progress:
        print("[annular] extracting annulus intensities (reshape)...")
    annulus_vals = data4d[:, :, annulus_mask].reshape(Sy * Sx, M).astype(np.float32)

    if print_progress:
        print("[annular] summing per-angle-bin (matmul)...")
    bin_sums = annulus_vals @ bin_mat  # (Sy*Sx, angle_bins)

    if print_progress:
        print("[annular] computing argmax angle + total intensity...")
    argmax_bin = np.argmax(bin_sums, axis=1).astype(np.int32)
    annulus_total = np.sum(bin_sums, axis=1).astype(np.float32)

    if print_progress:
        print("[annular] normalizing brightness...")
    v = np.log1p(annulus_total) if log_intensity else annulus_total
    lo = np.percentile(v, vmin_percentile)
    hi = np.percentile(v, vmax_percentile)
    if hi <= lo:
        hi = lo + 1e-6
    v_norm = (min_value + (1.0 - min_value) *
              np.clip((v - lo) / (hi - lo), 0.0, 1.0)).astype(np.float32)

    if print_progress:
        print("[annular] mapping HSV->RGB...")
    h = (argmax_bin.astype(np.float32) + 0.5) / float(angle_bins)
    s = np.ones_like(h, dtype=np.float32)
    hsv = np.stack([h, s, v_norm], axis=1)
    rgb = hsv_to_rgb(hsv).reshape(Sy, Sx, 3).astype(np.float32)

    debug = {
        "center": (cy, cx),
        "r_inner": float(r_inner),
        "r_outer": float(r_outer),
        "angle_bins": int(angle_bins),
        "annulus_mask": annulus_mask,                    # (Ky,Kx)
        "annulus_total": annulus_total.reshape(Sy, Sx),  # (Sy,Sx)
        "argmax_bin": argmax_bin.reshape(Sy, Sx),        # (Sy,Sx)
        # NOTE: omit bin_sums by default to keep memory sane
    }

    fig_rgb_tuple = None
    fig_dp_tuple = None

    if show_plots:
        # --- Plot RGB with external colorwheel ---
        fig = plt.figure()
        ax_img = fig.add_subplot(1, 1, 1)
        ax_img.imshow(rgb)
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        ax_img.set_title(title or "Annular max-angle hue / annulus intensity value")

        # Make room for wheel
        if wheel_side.lower() == "right":
            fig.subplots_adjust(right=0.82)
            wheel_left = 0.84 + wheel_margin
        elif wheel_side.lower() == "left":
            fig.subplots_adjust(left=0.18)
            wheel_left = 0.02 + wheel_margin
        else:
            raise ValueError("wheel_side must be 'right' or 'left'")

        ax_w = fig.add_axes([wheel_left, 0.02 + wheel_margin, 0.14, 0.14])

        L = int(wheel_px)
        gy, gx = np.indices((L, L), dtype=np.float32)
        gc = (L - 1) / 2.0
        gdx = gx - gc
        gdy = gy - gc
        gr = np.sqrt(gdx * gdx + gdy * gdy)
        gtheta = np.arctan2(gdy, gdx)

        inside = gr <= gc
        gu = (gtheta + np.pi) / (2.0 * np.pi)
        gh = (gu % 1.0).astype(np.float32)
        gs = np.ones_like(gh, dtype=np.float32)
        gv = np.ones_like(gh, dtype=np.float32)

        wheel = np.zeros((L, L, 3), dtype=np.float32)
        wheel_inside = hsv_to_rgb(np.stack([gh, gs, gv], axis=-1))
        wheel[inside] = wheel_inside[inside]

        ax_w.imshow(wheel)
        ax_w.set_xticks([])
        ax_w.set_yticks([])
        ax_w.set_title("Angle", fontsize=9)

        fig_rgb_tuple = (fig, ax_img, ax_w)

        # --- Plot one DP with annulus mask overlay (optional) ---
        if dp_index is None:
            dp_index = (Sy // 2, Sx // 2)
        sy0, sx0 = dp_index

        dp = data4d[sy0, sx0].astype(np.float32)
        show = np.log1p(dp) if log_dp else dp

        fig2, ax2 = plt.subplots(1, 1)
        ax2.imshow(show, cmap="gray")
        ax2.set_title(f"DP @ (sy={sy0}, sx={sx0}) with annulus mask")
        ax2.set_xticks([])
        ax2.set_yticks([])

        if dp_index is None:
            dp_index = (Sy // 2, Sx // 2)
        sy0, sx0 = dp_index

        dp = data4d[sy0, sx0].astype(np.float32)
        show = np.log1p(dp) if log_dp else dp

        fig2, ax2 = plt.subplots(1, 1)
        ax2.imshow(show, cmap="gray")
        ax2.set_title(f"DP @ (sy={sy0}, sx={sx0}) with annulus segments")
        ax2.set_xticks([])
        ax2.set_yticks([])

        # Build a per-pixel bin index map on the detector for visualization
        # (outside annulus = -1)
        u_full = (theta + np.pi) / (2.0 * np.pi)  # [0,1)
        seg_map = np.full((Ky, Kx), -1, dtype=np.int32)
        seg_map[annulus_mask] = np.clip(np.floor(u_full[annulus_mask] * angle_bins).astype(np.int32), 0, angle_bins - 1)

        # Color each segment by its hue (same mapping as the scan image)
        h = (seg_map.astype(np.float32) + 0.5) / float(angle_bins)
        h = np.mod(h, 1.0)
        s = np.ones_like(h, dtype=np.float32)
        v = np.ones_like(h, dtype=np.float32)
        seg_rgb = hsv_to_rgb(np.stack([h, s, v], axis=-1)).astype(np.float32)

        # RGBA overlay: only show within annulus
        seg_rgba = np.zeros((Ky, Kx, 4), dtype=np.float32)
        seg_rgba[..., :3] = seg_rgb
        seg_rgba[..., 3] = mask_alpha * (seg_map >= 0).astype(np.float32)

        ax2.imshow(seg_rgba)

        # Draw inner/outer circles for clarity
        ax2.add_patch(Circle((cx, cy), r_inner, fill=False, linewidth=1.2))
        ax2.add_patch(Circle((cx, cy), r_outer, fill=False, linewidth=1.2))

        fig_dp_tuple = (fig2, ax2)

        plt.show()

    if print_progress:
        print("[annular] wrapper complete.")

    return rgb, debug, fig_rgb_tuple, fig_dp_tuple

