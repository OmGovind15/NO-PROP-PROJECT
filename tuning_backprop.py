import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import time
from scipy.optimize import linear_sum_assignment
import torchvision.ops as ops
import matplotlib.pyplot as plt
from torchvision.datasets import VOCDetection
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from PIL import Image
import optuna


try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterator):
        return iterator



class FlowMatchingBlock(nn.Module):
    """Flow matching block that predicts vector field."""
    def __init__(self, d_model, nhead, num_queries):
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries
        self.self_attn = MultiHeadAttention(d_model, nhead)
        self.cross_attn = MultiHeadAttention(d_model, nhead)
        self.time_proj = nn.Linear(d_model, d_model)
        self.ffn = FeedForward(d_model, d_model * 2)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        self.vector_field_proj = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model))

    def forward(self, z_t, memory, pos_encoding, time_emb):
        time_cond = self.time_proj(time_emb).unsqueeze(1).expand(-1, self.num_queries, -1)
        z_t_cond = self.norm1(z_t + time_cond)
        attn_out = self.self_attn(z_t_cond.permute(1, 0, 2), z_t_cond.permute(1, 0, 2), z_t_cond.permute(1, 0, 2)).permute(1, 0, 2)
        z_t = self.norm2(z_t + self.dropout(attn_out))
        cross_attn_out = self.cross_attn(z_t.permute(1, 0, 2), memory + pos_encoding, memory + pos_encoding).permute(1, 0, 2)
        z_t = self.norm3(z_t + self.dropout(cross_attn_out))
        ffn_out = self.ffn(z_t.permute(1, 0, 2)).permute(1, 0, 2)
        z_t = self.norm4(z_t + self.dropout(ffn_out))
        vector_field = self.vector_field_proj(z_t)
        return vector_field

# --- NEW STANDARD METHOD COMPONENTS ---

class StandardTransformerDecoderLayer(nn.Module):
    """ A standard Transformer Decoder Layer without time conditioning. """
    def __init__(self, d_model, nhead):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead)
        self.cross_attn = MultiHeadAttention(d_model, nhead)
        self.ffn = FeedForward(d_model, d_model * 2)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, tgt, memory, pos_encoding):
        # Self-attention on the queries (tgt)
        attn_out = self.self_attn(tgt.permute(1, 0, 2), tgt.permute(1, 0, 2), tgt.permute(1, 0, 2)).permute(1, 0, 2)
        tgt = self.norm1(tgt + self.dropout(attn_out))
        
        # Cross-attention between queries (tgt) and image features (memory)
        cross_attn_out = self.cross_attn(tgt.permute(1, 0, 2), memory + pos_encoding, memory + pos_encoding).permute(1, 0, 2)
        tgt = self.norm2(tgt + self.dropout(cross_attn_out))

        # Feed-forward network
        ffn_out = self.ffn(tgt.permute(1, 0, 2)).permute(1, 0, 2)
        tgt = self.norm3(tgt + self.dropout(ffn_out))
        
        return tgt

class StandardEndToEndDetector(nn.Module):
    """ A standard end-to-end detector like DETR. """
    def __init__(self, num_classes, num_queries, d_model, nhead, num_decoder_layers):
        super().__init__()
        self.num_queries = num_queries
        self.d_model = d_model
        
        self.num_classes = num_classes
        # --- END FIX ---
        
        # Backbone and projection
        resnet_backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet_backbone.children())[:-2])
        self.projection = nn.Conv2d(2048, d_model, kernel_size=1)
        
        self.pos_encoder = PositionalEncoding(d_model)

        # Learned object queries
        self.query_embed = nn.Embedding(num_queries, d_model)
        
        # Decoder stack
        self.decoder_layers = nn.ModuleList([
            StandardTransformerDecoderLayer(d_model, nhead)
            for _ in range(num_decoder_layers)
        ])

        # Prediction heads
        self.class_head = nn.Linear(d_model, num_classes + 1)
        self.bbox_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 4)
        )

    def forward(self, images):
        # 1. Get image features from the backbone
        features = self.backbone(images)
        proj_features = self.projection(features)
        memory = proj_features.flatten(2).permute(2, 0, 1)
        pos_encoding = self.pos_encoder(torch.zeros_like(memory))
        
        # 2. Get initial object queries
        # Shape: (batch_size, num_queries, d_model)
        queries = self.query_embed.weight.unsqueeze(0).repeat(images.size(0), 1, 1)
        
        # 3. Pass queries through the decoder stack
        for layer in self.decoder_layers:
            queries = layer(queries, memory, pos_encoding)
            
        # 4. Get final predictions
        logits = self.class_head(queries)
        boxes = self.bbox_head(queries).sigmoid()
        
        return {'pred_logits': logits, 'pred_boxes': boxes}

    def inference(self, images):
        # For this model, inference is the same as the forward pass
        return self.forward(images)
# --- NEW STANDARD TRAINING LOOP ---

def train_standard_end_to_end(model, data_loader, val_loader, criterion, epochs, device, lr, lr_backbone, trial=None): # MODIFICATION 1: Add trial=None
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {"params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad], "lr": lr_backbone},
    ]
    optimizer = optim.AdamW(param_dicts, lr=lr, weight_decay=1e-4)
    model.to(device)
    history = []

    for epoch in range(epochs):
        model.train()
        epoch_loss_sum = 0
        num_batches = 0
        progress_bar = tqdm(data_loader, leave=True)

        for images, targets in progress_bar:
            images = torch.stack(images).to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(images)
            detection_losses = criterion(outputs, targets)
            weight_dict = {'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2}
            total_loss = sum(detection_losses[k] * weight_dict[k] for k in detection_losses.keys() if k in weight_dict)
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()
            epoch_loss_sum += total_loss.item()
            num_batches += 1
            progress_bar.set_postfix({'Loss': f'{total_loss.item():.4f}'})

        avg_epoch_loss = epoch_loss_sum / num_batches
        val_map = calculate_map(model, val_loader, device, model.num_classes)
        history.append({'epoch': epoch+1, 'mAP': val_map, 'loss': avg_epoch_loss})
        print(f"Epoch {epoch+1} Validation mAP: {val_map:.4f} | Avg Epoch Loss: {avg_epoch_loss:.4f}")

       
        if trial:
            trial.report(val_map, epoch)
            if trial.should_prune():
                print("--- Trial pruned ---")
                raise optuna.exceptions.TrialPruned()

    return history


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer('pe', pe.unsqueeze(0).transpose(0, 1))

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.nhead, self.d_model, self.head_dim = nhead, d_model, d_model//nhead
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        bs = query.size(1)
        q, k, v = self.q_linear(query), self.k_linear(key), self.v_linear(value)
        q = q.view(query.size(0), bs * self.nhead, self.head_dim).transpose(0, 1)
        k = k.view(key.size(0), bs * self.nhead, self.head_dim).transpose(0, 1)
        v = v.view(value.size(0), bs * self.nhead, self.head_dim).transpose(0, 1)
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        context = torch.bmm(torch.softmax(scores, dim=-1), v)
        context = context.transpose(0, 1).contiguous().view(query.size(0), bs, self.d_model)
        return self.out_linear(context)

class FeedForward(nn.Module):
    def __init__(self, d, dim_ff=2048, drop=0.1):
        super().__init__()
        self.l1 = nn.Linear(d, dim_ff)
        self.drop = nn.Dropout(drop)
        self.l2 = nn.Linear(dim_ff, d)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.l2(self.drop(self.relu(self.l1(x))))

class PascalVOCDatasetWrapper(Dataset):
    def __init__(self, root, image_set, download=True):
        self.voc_dataset = VOCDetection(
            root=root,
            year='2007',
            image_set=image_set,
            download=download,
            transform=transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        )
        self.voc_classes = [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
            'train', 'tvmonitor'
        ]
        self.class_to_idx = {cls: i for i, cls in enumerate(self.voc_classes)}

    def __len__(self):
        return len(self.voc_dataset)

    def __getitem__(self, idx):
        img_tensor, target_raw = self.voc_dataset[idx]
        boxes = []
        labels = []
        
        if 'size' in target_raw['annotation']:
            img_width = int(target_raw['annotation']['size']['width'])
            img_height = int(target_raw['annotation']['size']['height'])
        else:
            img_width = 1 
            img_height = 1

        if 'object' in target_raw['annotation']:
            objects = target_raw['annotation']['object']
            if not isinstance(objects, list):
                objects = [objects]
            
            for obj in objects:
                class_name = obj['name']
                if class_name in self.class_to_idx:
                    xmin = float(obj['bndbox']['xmin'])
                    ymin = float(obj['bndbox']['ymin'])
                    xmax = float(obj['bndbox']['xmax'])
                    ymax = float(obj['bndbox']['ymax'])
                    cx = ((xmin + xmax) / 2) / img_width
                    cy = ((ymin + ymax) / 2) / img_height
                    w = (xmax - xmin) / img_width
                    h = (ymax - ymin) / img_height
                    boxes.append([cx, cy, w, h])
                    labels.append(self.class_to_idx[class_name])

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32).clamp(min=0, max=1),
            'labels': torch.tensor(labels, dtype=torch.int64)
        }
        return img_tensor, target

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def generalized_box_iou(boxes1, boxes2):
    boxes1 = box_cxcywh_to_xyxy(boxes1)
    boxes2 = box_cxcywh_to_xyxy(boxes2)
    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])
    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    b1_area = (boxes1[:, 2]-boxes1[:, 0])*(boxes1[:, 3]-boxes1[:, 1])
    b2_area = (boxes2[:, 2]-boxes2[:, 0])*(boxes2[:, 3]-boxes2[:, 1])
    union_area = b1_area[:, None] + b2_area - inter_area
    iou = inter_area / (union_area + 1e-6)
    enclose_x1 = torch.min(boxes1[:, None, 0], boxes2[:, 0])
    enclose_y1 = torch.min(boxes1[:, None, 1], boxes2[:, 1])
    enclose_x2 = torch.max(boxes1[:, None, 2], boxes2[:, 2])
    enclose_y2 = torch.max(boxes1[:, None, 3], boxes2[:, 3])
    enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1)
    return iou - (enclose_area - union_area) / (enclose_area + 1e-6)

class HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 1, cost_bbox: float = 5, cost_giou: float = 2):
        super().__init__()
        self.cost_class, self.cost_bbox, self.cost_giou = cost_class, cost_bbox, cost_giou

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        out_bbox = outputs["pred_boxes"].flatten(0, 1)
        tgt_ids = torch.cat([t["labels"] for t in targets])
        tgt_bbox = torch.cat([t["boxes"] for t in targets])
        cost_class = -out_prob[:, tgt_ids]
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_giou = -generalized_box_iou(out_bbox, tgt_bbox)
        C = (self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou).view(bs, num_queries, -1).cpu()
        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, eos_coef):
        super().__init__()
        self.num_classes, self.matcher, self.eos_coef = num_classes, matcher, eos_coef
        weight = torch.ones(self.num_classes + 1)
        weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', weight)
        self.ce_loss = nn.CrossEntropyLoss(weight=self.empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(outputs['pred_logits'].shape[:2], self.num_classes, dtype=torch.int64, device=outputs['pred_logits'].device)
        target_classes[idx] = target_classes_o
        return {"loss_ce": self.ce_loss(outputs['pred_logits'].transpose(1, 2), target_classes)}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_bbox = nn.functional.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_giou = 1 - torch.diag(generalized_box_iou(src_boxes, target_boxes))
        return {"loss_bbox": loss_bbox.sum() / num_boxes, "loss_giou": loss_giou.sum() / num_boxes}

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def forward(self, outputs, targets):
        indices = self.matcher(outputs, targets)
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if num_boxes.item() == 0:
            return {
                'loss_ce': torch.tensor(0.0, device=num_boxes.device),
                'loss_bbox': torch.tensor(0.0, device=num_boxes.device),
                'loss_giou': torch.tensor(0.0, device=num_boxes.device)
            }
        losses = {}
        losses.update(self.loss_labels(outputs, targets, indices, num_boxes))
        losses.update(self.loss_boxes(outputs, targets, indices, num_boxes))
        return losses

def calculate_map(model, data_loader, device, num_classes, conf_threshold=0.5, nms_threshold=0.5):
    model.eval()
    dataset_pred_boxes = []
    dataset_pred_scores = []
    dataset_pred_labels = []
    dataset_true_boxes = []
    dataset_true_labels = []
    dataset_true_img_idx = []

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(data_loader, desc="Calculating mAP")):
            images = torch.stack(images).to(device)
            outputs = model.inference(images)
            
            probs = outputs['pred_logits'].softmax(-1).cpu()
            boxes_cxcywh = outputs['pred_boxes'].cpu()

            for i in range(len(targets)):
                scores, labels = probs[i].max(-1)
                keep = scores > conf_threshold
                if not keep.any():
                    continue
                
                filtered_boxes = box_cxcywh_to_xyxy(boxes_cxcywh[i][keep])
                filtered_scores = scores[keep]
                filtered_labels = labels[keep] % num_classes

                nms_indices = ops.batched_nms(filtered_boxes, filtered_scores, filtered_labels, nms_threshold)
                
                dataset_pred_boxes.append(filtered_boxes[nms_indices])
                dataset_pred_scores.append(filtered_scores[nms_indices])
                dataset_pred_labels.append(filtered_labels[nms_indices])
                
                true_boxes = targets[i]['boxes']
                if len(true_boxes) > 0:
                    dataset_true_boxes.append(box_cxcywh_to_xyxy(true_boxes))
                    dataset_true_labels.append(targets[i]['labels'])
                    dataset_true_img_idx.append(torch.full_like(targets[i]['labels'], fill_value=batch_idx * len(targets) + i))

    if not dataset_pred_boxes:
        return 0.0
        
    preds = {
        'boxes': torch.cat(dataset_pred_boxes, dim=0),
        'scores': torch.cat(dataset_pred_scores, dim=0),
        'labels': torch.cat(dataset_pred_labels, dim=0)
    }
    if not dataset_true_boxes:
        return 0.0

    truths = {
        'boxes': torch.cat(dataset_true_boxes, dim=0),
        'labels': torch.cat(dataset_true_labels, dim=0),
        'img_idx': torch.cat(dataset_true_img_idx, dim=0)
    }
    
    return mean_average_precision_vectorized(preds, truths, num_classes=num_classes)

def mean_average_precision_vectorized(pred_boxes, true_boxes, iou_threshold=0.5, num_classes=20):
    average_precisions = []
    sorted_indices = torch.argsort(pred_boxes['scores'], descending=True)
    
    for c in range(num_classes):
        class_preds_mask = (pred_boxes['labels'][sorted_indices] == c)
        class_preds_indices = sorted_indices[class_preds_mask]
        class_truths_mask = (true_boxes['labels'] == c)

        if not class_truths_mask.any():
            if class_preds_mask.any():
                average_precisions.append(torch.tensor(0.0))
            continue
            
        if not class_preds_mask.any():
            average_precisions.append(torch.tensor(0.0))
            continue

        num_class_preds = len(class_preds_indices)
        num_class_truths = class_truths_mask.sum()
        TP = torch.zeros(num_class_preds)
        FP = torch.zeros(num_class_preds)
        gt_img_indices = true_boxes['img_idx'][class_truths_mask]
        gt_matched = torch.zeros_like(gt_img_indices, dtype=torch.bool)

        iou_matrix = ops.box_iou(
            pred_boxes['boxes'][class_preds_indices], 
            true_boxes['boxes'][class_truths_mask]
        )

        for i, pred_idx in enumerate(class_preds_indices):
            best_iou, best_gt_idx = iou_matrix[i].max(0)
            if best_iou > iou_threshold:
                if not gt_matched[best_gt_idx]:
                    TP[i] = 1
                    gt_matched[best_gt_idx] = True
                else:
                    FP[i] = 1
            else:
                FP[i] = 1

        TP_cumsum = torch.cumsum(TP, 0)
        FP_cumsum = torch.cumsum(FP, 0)
        recalls = TP_cumsum / (num_class_truths + 1e-6)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + 1e-6)
        precisions = torch.cat((torch.tensor([1.0]), precisions))
        recalls = torch.cat((torch.tensor([0.0]), recalls))
        average_precisions.append(torch.trapz(precisions, recalls))

    return (torch.mean(torch.stack(average_precisions))).item() if average_precisions else 0.0
    
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
# --- MAIN EXECUTION BLOCK (MODIFIED FOR OPTUNA) ---


def objective(trial):
    """
    This function defines a single trial for Optuna.
    """
    # 1. Suggest hyperparameters, including the new batch_size list
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    lr_backbone = trial.suggest_float("lr_backbone", 1e-6, 1e-4, log=True)
    n_head = trial.suggest_categorical("n_head", [4, 8])
    num_decoder_layers = trial.suggest_int("num_decoder_layers", 4, 12)
    
    # MODIFIED: Using your requested batch sizes
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 12])

    # Static Hyperparameters
    D_MODEL, NUM_QUERIES, NUM_CLASSES = 128, 20, 20
    EPOCHS_FOR_SEARCH = 15

    print(f"\n--- Starting Trial {trial.number} ---")
    print(f"Params: lr={lr:.6f}, lr_backbone={lr_backbone:.6f}, n_head={n_head}, dec_layers={num_decoder_layers}, batch_size={batch_size}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    # 2. Handle Potential Memory Errors
    try:
        # Data loaders now use the suggested batch size
        root_path = '/home/prajna/Downloads/pascal_voc_project'
        dataset = PascalVOCDatasetWrapper(root=root_path, image_set='trainval', download=False)
        val_dataset = PascalVOCDatasetWrapper(root=root_path, image_set='test', download=False)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)), num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=lambda x: tuple(zip(*x)), num_workers=0)

        # Set up the model and criterion
        model = StandardEndToEndDetector(NUM_CLASSES, NUM_QUERIES, D_MODEL, n_head, num_decoder_layers)
        matcher = HungarianMatcher()
        criterion = SetCriterion(NUM_CLASSES, matcher, 0.1).to(device)

        # Run the training
        history = train_standard_end_to_end(model, data_loader, val_loader, criterion, EPOCHS_FOR_SEARCH, device, lr=lr, lr_backbone=lr_backbone, trial=trial)

        # Return the final score
        final_map = history[-1]['mAP'] if history else 0.0
        return final_map

    except RuntimeError as e:
        # Catch CUDA out-of-memory errors
        if "out of memory" in str(e):
            print(f"--- Trial pruned due to Out Of Memory with batch_size={batch_size} ---")
            # Prune the trial to tell Optuna this set of params is invalid
            raise optuna.exceptions.TrialPruned()
        else:
            # Re-raise other runtime errors
            raise e


if __name__ == '__main__':
    # This part remains the same
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=30) 

    print("\n" + "="*50 + "\n     OPTUNA SEARCH COMPLETE     \n" + "="*50)
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    best_trial = study.best_trial
    print(f"  Value (mAP): {best_trial.value:.4f}")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

