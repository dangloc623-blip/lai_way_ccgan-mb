# Dedicated CycleGAN Architecture & Developer Guide

Tài liệu hướng dẫn kỹ thuật toàn diện cho dự án **CycleGAN Tinh Gọn (Hỗ trợ cả Unpaired và Paired Datasets)**.

---

## 📑 Mục lục
1. [Tổng quan dự án & Cấu trúc thư mục](#1-tổng-quan-dự-án--cấu-trúc-thư-mục)
2. [Chi tiết chức năng của từng file](#2-chi-tiết-chức-năng-của-từng-file)
   - [2.1. Thư mục gốc (Root)](#21-thư-mục-gốc-root)
   - [2.2. Thư mục `models/`](#22-thư-mục-models)
   - [2.3. Thư mục `data/`](#23-thư-mục-data)
   - [2.4. Thư mục `options/`](#24-thư-mục-options)
   - [2.5. Thư mục `util/`](#25-thư-mục-util)
   - [2.6. Thư mục `datasets/`](#26-thư-mục-datasets)
3. [Luồng hoạt động & Đồ thị tính toán (Dataflow & Losses)](#3-luồng-hoạt-động--đồ-thị-tính-toán)
4. [Cẩm nang kỹ thuật: Hướng dẫn chỉnh sửa & Mở rộng Model](#4-cẩm-nang-kỹ-thuật-hướng-dẫn-chỉnh-sửa--mở-rộng-model)
   - [Hướng dẫn 1: Thêm một hàm Loss mới (SSIM, Perceptual/VGG, L1 Paired)](#hướng-dẫn-1-thêm-một-hàm-loss-mới)
   - [Hướng dẫn 2: Thay đổi / Tùy biến kiến trúc Generator (Attention, CBAM, UNet)](#hướng-dẫn-2-thay-đổi--tùy-biến-kiến-trúc-generator)
   - [Hướng dẫn 3: Tùy biến kiến trúc Discriminator (Patch size, Spectral Norm)](#hướng-dẫn-3-tùy-biến-kiến-trúc-discriminator)
   - [Hướng dẫn 4: Huấn luyện với Dữ liệu có cặp (Paired Dataset)](#hướng-dẫn-4-huấn-luyện-với-dữ-liệu-có-cặp-paired-dataset)
   - [Hướng dẫn 5: Thay đổi số kênh & Tiền xử lý dữ liệu (Ảnh Grayscale / Y tế)](#hướng-dẫn-5-thay-đổi-số-kênh--tiền-xử-lý-dữ-liệu)
   - [Hướng dẫn 6: Theo dõi thêm ảnh trung gian lên trang HTML](#hướng-dẫn-6-theo-dõi-thêm-ảnh-trung-gian-lên-trang-html)
   - [Hướng dẫn 7: Thay đổi Optimizer và Lịch trình Learning Rate](#hướng-dẫn-7-thay-đổi-optimizer-và-lịch-trình-learning-rate)
5. [Hướng dẫn cài đặt & Chạy thực nghiệm](#5-hướng-dẫn-cài-đặt--chạy-thực-nghiệm)
6. [Quy tắc ngầm (Gotchas) & Lưu ý quan trọng khi lập trình](#6-quy-tắc-ngầm-gotchas--lưu-ý-quan-trọng-khi-lập-trình)

---

## 1. Tổng quan dự án & Cấu trúc thư mục

```text
CycleGAN_Project/
│
├── train.py                  # Script điều phối toàn bộ quá trình huấn luyện
├── test.py                   # Script thực thi kiểm thử và sinh ảnh (Inference)
├── environment.yml           # Khai báo môi trường Conda và các thư viện cần thiết
├── README.md                 # Tài liệu hướng dẫn kỹ thuật toàn diện của dự án
│
├── models/                   # Kiến trúc mô hình mạng và thuật toán tối ưu
│   ├── __init__.py           # Dynamic importer tự động nạp class Model
│   ├── base_model.py         # Lớp cha trừu tượng quản lý GPU, Checkpoints, Schedulers
│   ├── cycle_gan_model.py    # Triển khai thuật toán CycleGAN (Forward, Backward, Loss)
│   ├── networks.py           # Định nghĩa ResNet, UNet, PatchGAN, GANLoss, Schedulers
│   └── test_model.py         # Model chuyên dụng để test 1 chiều nhanh (A -> B)
│
├── data/                     # Data pipeline và tiền xử lý
│   ├── __init__.py           # Dynamic importer nạp Dataset và cấu hình DataLoader
│   ├── base_dataset.py       # Lớp cha BaseDataset và các hàm transform tiền xử lý
│   ├── image_folder.py       # Quét đệ quy đường dẫn ảnh từ ổ đĩa
│   ├── aligned_dataset.py    # Dataset nạp dữ liệu CÓ CẶP (Ảnh ghép đôi [A | B])
│   └── single_dataset.py     # Dataset nạp 1 thư mục ảnh đơn lẻ cho inference
│
├── options/                  # Parser cấu hình siêu tham số (Hyperparameters)
│   ├── __init__.py           # Package init
│   ├── base_options.py       # Cấu hình dùng chung (GPU, I/O paths, architecture, preproc)
│   ├── train_options.py      # Cấu hình riêng cho huấn luyện (LR, epoch, loss weights)
│   └── test_options.py       # Cấu hình riêng cho kiểm thử (results dir, checkpoint epoch)
│
└── util/                     # Các công cụ phụ trợ
    ├── __init__.py           # Package init
    ├── image_pool.py         # Buffer lưu trữ 50 ảnh fake giúp ổn định Discriminator
    ├── util.py               # Chuyển đổi Tensor <-> Numpy/PIL, khởi tạo DDP, tạo thư mục
    ├── visualizer.py         # Ghi log loss, lưu ảnh trực quan hóa qua HTML / Visdom
    └── html.py               # Engine tạo trang web tĩnh xem kết quả trực quan


```

---

## 2. Chi tiết chức năng của từng file

### 2.1. Thư mục gốc (Root)
* **`train.py`**:
  * Điểm bắt đầu (entry-point) của quá trình huấn luyện.
  * Khởi tạo `TrainOptions`, thiết lập GPU/DDP, tạo `dataset` và `model`.
  * Thực hiện vòng lặp `epoch` và `iteration`, gọi `model.set_input(data)` $\to$ `model.optimize_parameters()`.
  * Gọi `visualizer` để log loss ra terminal/file và định kỳ gọi `model.save_networks()`.
* **`test.py`**:
  * Điểm bắt đầu của quá trình kiểm thử / suy luận.
  * Tự động tắt tính toán gradient (`torch.no_grad()`), nạp trọng số checkpoint đã lưu, chạy suy luận qua `model.test()`.
  * Lưu kết quả ảnh sinh ra và render trang web HTML tổng hợp bằng `util.html`.
* **`environment.yml`**:
  * File đặc tả môi trường Conda (PyTorch, TorchVision, Pillow, Visdom, v.v.) giúp tái lập môi trường chạy nhất quán.

---

### 2.2. Thư mục `models/`
* **`models/__init__.py`**:
  * Cung cấp hàm `create_model(opt)` và `find_model_using_name(model_name)`.
  * Quét thư mục `models/` và nạp động module `[model_name]_model.py` (ví dụ `--model cycle_gan` $\to$ nạp class `CycleGANModel`).
* **`models/base_model.py`**:
  * Lớp cơ sở trừu tượng (`BaseModel`).
  * Quản lý chuyển mô hình lên CPU/CUDA (`opt.device`).
  * Tự động duyệt qua danh sách `self.model_names` để lưu (`save_networks`) và nạp (`load_networks`) file `.pth`.
  * Tự động thu thập giá trị loss từ danh sách `self.loss_names` qua hàm `get_current_losses()`.
  * Tự động trích xuất các tensor ảnh cần hiển thị từ `self.visual_names` qua hàm `get_current_visuals()`.
  * Quản lý `update_learning_rate()` theo từng epoch.
* **`models/cycle_gan_model.py`**:
  * **Trọng tâm của CycleGAN**: Kế thừa `BaseModel`.
  * Khởi tạo 2 Generator ($G_A: A \to B$, $G_B: B \to A$) và 2 Discriminator ($D_A, D_B$).
  * Quản lý `ImagePool` cho ảnh giả `fake_A` và `fake_B`.
  * `forward()`: Thực hiện sinh ảnh $fake\_B = G_A(real\_A)$, $rec\_A = G_B(fake\_B)$ và ngược lại.
  * `backward_G()`: Tính tổng hợp Identity Loss, GAN Loss và Cycle-Consistency Loss.
  * `backward_D_A()` / `backward_D_B()`: Tính loss cho các bộ phân biệt $D_A, D_B$.
  * `optimize_parameters()`: Điều phối bật/tắt gradient và bước nhảy của `optimizer_G` và `optimizer_D`.
* **`models/networks.py`**:
  * Chứa toàn bộ các khối kiến trúc mạng PyTorch (`nn.Module`):
    * `ResnetGenerator`: Generator nền tảng của CycleGAN với 6 hoặc 9 Residual Blocks.
    * `UnetGenerator` & `UnetSkipConnectionBlock`: Generator kiểu U-Net với các kết nối tắt (skip connections).
    * `NLayerDiscriminator`: PatchGAN Discriminator (mặc định receptive field $70 \times 70$).
    * `PixelDiscriminator`: $1 \times 1$ PatchGAN phân loại từng pixel.
    * `GANLoss`: Lớp trừu tượng hóa nhãn thật/giả và tính loss (LSGAN với MSE, Vanilla với BCE, WGAN-GP).
    * `get_scheduler()`: Tạo scheduler giảm tốc độ học (`linear`, `step`, `plateau`, `cosine`).
* **`models/test_model.py`**:
  * Model rút gọn chỉ nạp duy nhất một Generator $G$ để suy luận nhanh 1 chiều (tiết kiệm bộ nhớ GPU).

---

### 2.3. Thư mục `data/`
* **`data/__init__.py`**:
  * Cung cấp `create_dataset(opt)` và class wrapper `CustomDatasetDataLoader`.
  * Quét thư mục `data/` và nạp động dataset dựa trên cờ `--dataset_mode` (ví dụ: `unaligned` hoặc `aligned`).
* **`data/base_dataset.py`**:
  * Lớp cơ sở `BaseDataset`.
  * Định nghĩa hàm `get_transform()` chuẩn hóa ảnh: resize, crop, lật ngang ngẫu nhiên (flip), chuyển sang Tensor và scale giá trị về $[-1, 1]$.
* **`data/image_folder.py`**:
  * Chứa hàm `make_dataset()` hỗ trợ quét tất cả các file có đuôi ảnh hợp lệ (`.jpg`, `.png`, `.tif`, v.v.) trong các thư mục con.
* **`data/aligned_dataset.py`**:
  * **Dataset dành cho Dữ liệu có cặp (Paired)**: Đọc từng file ảnh đơn đã được ghép đôi sẵn theo chiều ngang $[A \mid B]$.
  * Hàm `__getitem__` tự động cắt đôi ảnh thành 2 phần bằng nhau: nửa trái là `A`, nửa phải là `B`.
* **`data/single_dataset.py`**:
  * Chỉ nạp ảnh từ 1 thư mục duy nhất, sử dụng khi chạy `test.py` với `--model test`.

---

### 2.4. Thư mục `options/`
* **`options/base_options.py`**:
  * Chứa class `BaseOptions`: Định nghĩa các tham số dùng chung cho cả train và test:
    * `--dataroot`: Đường dẫn thư mục dữ liệu.
    * `--name`: Tên thử nghiệm (dùng để đặt tên folder checkpoint và kết quả).
    * `--gpu_ids`: Chỉ định GPU (`0`, `0,1`, hoặc `-1` cho CPU).
    * `--model`: Chọn mô hình (`cycle_gan`, `test`).
    * `--dataset_mode`: Chọn loại dữ liệu (`unaligned`, `aligned`, `single`).
    * `--netG`, `--netD`, `--norm`, `--input_nc`, `--output_nc`, `--load_size`, `--crop_size`.
* **`options/train_options.py`**:
  * Chứa class `TrainOptions`: Kế thừa `BaseOptions`, thêm các cờ phục vụ huấn luyện:
    * `--n_epochs`, `--n_epochs_decay`: Số epoch giữ nguyên LR và số epoch giảm dần LR về 0.
    * `--lr`, `--beta1`: Cấu hình cho Adam Optimizer.
    * `--save_epoch_freq`, `--display_freq`, `--print_freq`.
* **`options/test_options.py`**:
  * Chứa class `TestOptions`: Kế thừa `BaseOptions`, thêm các cờ phục vụ kiểm thử:
    * `--results_dir`: Thư mục lưu ảnh sinh ra.
    * `--epoch`: Chọn checkpoint epoch nào để test (mặc định: `latest`).
    * `--num_test`: Số lượng ảnh tối đa cần suy luận.

---

### 2.5. Thư mục `util/`
* **`util/image_pool.py`**:
  * Cài đặt cơ chế **History Buffer**: Lưu trữ 50 ảnh giả được sinh ra gần nhất.
  * Giúp Discriminator không bị hiện tượng "quên" (catastrophic forgetting) và ngăn chặn việc Generator chỉ lừa được Discriminator ở iteration hiện tại.
* **`util/util.py`**:
  * Cung cấp các hàm tiện ích: `tensor2im()` (chuyển PyTorch tensor về numpy image để hiển thị), `save_image()`, `mkdirs()`, cấu hình môi trường Distributed Data Parallel (DDP).
* **`util/visualizer.py`**:
  * Theo dõi tiến trình: In loss ra console, ghi ra file `loss_log.txt`, vẽ đồ thị loss lên Visdom/WandB, và lưu bảng kết quả ảnh ra HTML.
* **`util/html.py`**:
  * Quản lý tạo file HTML tĩnh `index.html` với cấu trúc bảng hiển thị so sánh: Real A $\to$ Fake B $\to$ Rec A.

---

## 3. Luồng hoạt động & Đồ thị tính toán

### 3.1. Sơ đồ Forward và Chu trình khép kín
```
[Miền A: real_A] ---> ( Generator G_A ) ---> [ fake_B ] ---> ( Generator G_B ) ---> [ rec_A ]
        |                                       |
        |                                       v
        |                               ( Discriminator D_A )
        |                                       |
        v                                       v
 [ Loss Cycle A: ||rec_A - real_A||_1 ]    [ Loss GAN G_A: D_A(fake_B) ~ 1 ]

-----------------------------------------------------------------------------------------

[Miền B: real_B] ---> ( Generator G_B ) ---> [ fake_A ] ---> ( Generator G_A ) ---> [ rec_B ]
        |                                       |
        |                                       v
        |                               ( Discriminator D_B )
        |                                       |
        v                                       v
 [ Loss Cycle B: ||rec_B - real_B||_1 ]    [ Loss GAN G_B: D_B(fake_A) ~ 1 ]
```

### 3.2. Cơ chế hàm mục tiêu (Objective Losses)
Trong hàm `backward_G()`:
$$\mathcal{L}_{\text{total\_G}} = \mathcal{L}_{\text{GAN}}(G_A, D_A) + \mathcal{L}_{\text{GAN}}(G_B, D_B) + \lambda_A \mathcal{L}_{\text{cyc}}(G_A, G_B) + \lambda_B \mathcal{L}_{\text{cyc}}(G_B, G_A) + \lambda_{\text{idt}} \mathcal{L}_{\text{identity}}$$

Trong hàm `backward_D_A()` và `backward_D_B()`:
$$\mathcal{L}_{D} = \frac{1}{2} \left[ \mathbb{E}(\mathcal{L}_{\text{GAN}}(D(\text{real}), 1)) + \mathbb{E}(\mathcal{L}_{\text{GAN}}(D(\text{fake}.\text{detach}()), 0)) \right]$$

---

## 4. Cẩm nang kỹ thuật: Hướng dẫn chỉnh sửa & Mở rộng Model

Mục này cung cấp vị trí code chính xác và các bước cần làm đối với từng bài toán nghiên cứu thực tế.

---

### Hướng dẫn 1: Thêm một hàm Loss mới

Ví dụ: Bạn muốn bổ sung thêm **SSIM Loss** hoặc **Perceptual Loss** để nâng cao chất lượng ảnh sinh ra.

#### Bước 1: Khai báo trọng số loss trong CLI
Mở file `models/cycle_gan_model.py`, tìm hàm `modify_commandline_options`:
```python
@staticmethod
def modify_commandline_options(parser, is_train=True):
    # ... các dòng cũ ...
    if is_train:
        parser.add_argument('--lambda_ssim', type=float, default=5.0, help='trọng số cho SSIM loss')
    return parser
```

#### Bước 2: Đăng ký tên loss và khởi tạo hàm loss
Trong `models/cycle_gan_model.py`, tại hàm `__init__`:
```python
def __init__(self, opt):
    BaseModel.__init__(self, opt)
    # 1. Thêm tên hiển thị vào self.loss_names (BaseModel sẽ tự in ra console)
    self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'ssim_A', 'ssim_B']
    
    # 2. Khởi tạo hàm loss (nếu ở train mode)
    if self.isTrain:
        # Giả sử bạn có class SSIMLoss()
        from my_losses import SSIMLoss
        self.criterionSSIM = SSIMLoss().to(self.device)
```

#### Bước 3: Tính toán loss trong `backward_G()`
Trong `models/cycle_gan_model.py`, tìm hàm `backward_G()`:
```python
def backward_G(self):
    # ... tính các loss cũ ...
    
    # 3. Tính SSIM Loss cho cả 2 chiều
    if self.opt.lambda_ssim > 0:
        # Lưu ý quy tắc đặt tên: self.loss_<tên_trong_loss_names>
        self.loss_ssim_A = (1.0 - self.criterionSSIM(self.fake_B, self.real_B)) * self.opt.lambda_ssim
        self.loss_ssim_B = (1.0 - self.criterionSSIM(self.fake_A, self.real_A)) * self.opt.lambda_ssim
    else:
        self.loss_ssim_A = 0
        self.loss_ssim_B = 0

    # 4. Cộng dồn vào self.loss_G trước khi backward
    self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + \
                  self.loss_idt_A + self.loss_idt_B + self.loss_ssim_A + self.loss_ssim_B
    self.loss_G.backward()
```

---

### Hướng dẫn 2: Thay đổi / Tùy biến kiến trúc Generator

Nếu bạn muốn thay đổi ResNet bằng một kiến trúc mới (ví dụ: chèn cơ chế chú ý **CBAM**, **Self-Attention** hoặc cấu trúc Transformer bottleneck):

#### Bước 1: Viết class Generator mới trong `models/networks.py`
```python
class CustomAttentionGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d):
        super().__init__()
        # Định nghĩa các layer của bạn tại đây...
        
    def forward(self, x):
        # Forward pass...
        return out
```

#### Bước 2: Đăng ký mạng mới vào hàm `define_G`
Cũng trong `models/networks.py`, tìm hàm `define_G`:
```python
def define_G(input_nc, output_nc, ngf, netG, ...):
    # ...
    elif netG == 'custom_attn':
        net = CustomAttentionGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer)
    else:
        raise NotImplementedError(f"Generator model name [{netG}] is not recognized")
    return init_net(net, init_type, init_gain)
```

#### Bước 3: Chạy huấn luyện với Generator mới
Khi chạy lệnh train, chỉ cần thêm tham số:
```bash
python train.py --dataroot ./datasets/my_data --name exp_custom_gen --netG custom_attn
```

---

### Hướng dẫn 3: Tùy biến kiến trúc Discriminator

Mặc định CycleGAN dùng PatchGAN 70x70 (`NLayerDiscriminator` với `n_layers=3`).

* **Thay đổi kích thước PatchGAN**: 
  * Muốn trường tiếp nhận (Receptive field) lớn hơn (ví dụ patch size lớn hơn để bắt ngữ cảnh toàn cục): truyền cờ `--netD n_layers --n_layers_D 4` hoặc `5`.
  * Muốn phân loại ở cấp độ pixel $1 \times 1$: truyền cờ `--netD pixel`.
* **Thêm Spectral Normalization**:
  * Vào `models/networks.py`, tại class `NLayerDiscriminator`: Bọc các `nn.Conv2d` bằng `nn.utils.spectral_norm(nn.Conv2d(...))`.

---

### Hướng dẫn 4: Huấn luyện với Dữ liệu có cặp (Paired Dataset)

Nếu bạn có tập dữ liệu gồm các cặp ảnh tương ứng nhau (ví dụ: ảnh CT Không cản quang và Có cản quang của cùng một bệnh nhân):

#### Cách 1: Sử dụng `aligned_dataset.py` (Khuyên dùng)
1. **Ghép đôi ảnh**: Chạy script để ghép thư mục `trainA` và `trainB` thành ảnh ghép ngang $[A \mid B]$:
   ```bash
   python datasets/combine_A_and_B.py --fold_A /path/to/trainA --fold_B /path/to/trainB --fold_AB /path/to/trainAB
   ```
2. **Chạy train**:
   ```bash
   python train.py --dataroot /path/to/trainAB --name cyclegan_paired --dataset_mode aligned
   ```

#### Cách 2: Tận dụng dữ liệu có cặp để thêm Supervised L1 Loss
Khi có dữ liệu cặp, ngoài Cycle Loss, bạn có thể ép ảnh sinh ra `fake_B` phải giống hệt `real_B` theo L1 loss:
1. Trong `models/cycle_gan_model.py`, hàm `modify_commandline_options`:
   ```python
   parser.add_argument('--lambda_recon', type=float, default=100.0, help='trọng số L1 loss trực tiếp')
   ```
2. Trong `backward_G()`:
   ```python
   # Vì data có cặp nên real_B chính là ground-truth tương ứng của real_A!
   if hasattr(self.opt, 'lambda_recon') and self.opt.lambda_recon > 0:
       self.loss_recon_B = torch.nn.functional.l1_loss(self.fake_B, self.real_B) * self.opt.lambda_recon
       self.loss_G += self.loss_recon_B
   ```

---

### Hướng dẫn 5: Thay đổi số kênh & Tiền xử lý dữ liệu

* **Ảnh Grayscale (1 kênh) hoặc Ảnh Y tế chuyên dụng**:
  * Không cần sửa code, chỉ cần truyền cờ:
    ```bash
    python train.py --dataroot ./datasets/medical --input_nc 1 --output_nc 1
    ```
* **Tùy biến giải thuật Normalize/Augmentation**:
  * Vào file `data/base_dataset.py`, chỉnh sửa hàm `get_transform()`.
  * Nếu làm việc với ảnh CT (HU value) vượt quá khoảng $[0, 255]$, hãy chuyển việc đọc ảnh sang `SimpleITK` hoặc `tifffile` trong `__getitem__` của `data/aligned_dataset.py` hoặc `data/unaligned_dataset.py`.

---

### Hướng dẫn 6: Theo dõi thêm ảnh trung gian lên trang HTML

Nếu mô hình của bạn tính toán thêm một ảnh trung gian (ví dụ: ảnh Residual chênh lệch giữa Input và Output, hoặc Feature Map):

1. Trong `models/cycle_gan_model.py`, hàm `__init__`:
   ```python
   # Thêm tên biến vào visual_names
   self.visual_names.append('diff_A')
   ```
2. Trong hàm `forward()`:
   ```python
   # Tính toán và gán giá trị tensor cho thuộc tính diff_A
   self.diff_A = torch.abs(self.real_A - self.fake_B)
   ```
3. Hệ thống sẽ tự động chuyển đổi `self.diff_A` thành file ảnh và hiển thị lên file HTML báo cáo kết quả.

---

### Hướng dẫn 7: Thay đổi Optimizer và Lịch trình Learning Rate

* **Đổi sang AdamW**:
  * Mở `models/cycle_gan_model.py`, tìm đoạn khởi tạo optimizer trong `__init__`:
    ```python
    self.optimizer_G = torch.optim.AdamW(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=1e-4)
    self.optimizer_D = torch.optim.AdamW(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=1e-4)
    ```
* **Đổi Scheduler sang Cosine Annealing**:
  * Khi chạy train, chỉ cần thêm cờ:
    ```bash
    python train.py --lr_policy cosine --n_epochs 100
    ```

---

## 5. Hướng dẫn cài đặt & Chạy thực nghiệm

### 5.1. Khởi tạo môi trường
```bash
conda env create -f environment.yml
conda activate pytorch-CycleGAN-and-pix2pix
```

### 5.2. Huấn luyện (Training)

* **Huấn luyện với dữ liệu Unpaired (Mặc định)**:
  ```bash
  python train.py --dataroot ./datasets/horse2zebra --name h2z_cyclegan --model cycle_gan --dataset_mode unaligned --gpu_ids 0
  ```

* **Huấn luyện với dữ liệu Paired (Đã ghép đôi $[A \mid B]$)**:
  ```bash
  python train.py --dataroot ./datasets/my_paired_data --name paired_cyclegan --model cycle_gan --dataset_mode aligned --gpu_ids 0
  ```

* **Tiếp tục huấn luyện từ checkpoint trước đó (Resume Training)**:
  ```bash
  python train.py --dataroot ./datasets/my_data --name h2z_cyclegan --continue_train --epoch_count 51
  ```

### 5.3. Kiểm thử (Testing / Inference)
* **Test cả 2 chiều ($A \to B$ và $B \to A$)**:
  ```bash
  python test.py --dataroot ./datasets/horse2zebra --name h2z_cyclegan --model cycle_gan
  ```
* **Test nhanh 1 chiều duy nhất ($A \to B$)**:
  ```bash
  python test.py --dataroot ./datasets/horse2zebra/testA --name h2z_cyclegan --model test --no_dropout
  ```
* Kết quả ảnh và trang xem tĩnh sẽ được lưu tại: `./results/[tên_thử_nghiệm]/test_latest/index.html`.

---

## 6. Quy tắc ngầm (Gotchas) & Lưu ý quan trọng khi lập trình

> [!CAUTION]
> 1. **Rò rỉ Gradient trong Discriminator (`.detach()`)**:
>    Khi đưa ảnh fake vào Discriminator trong hàm `backward_D_basic()`, **bắt buộc** phải gọi `.detach()` (ví dụ: `netD(fake.detach())`). Nếu quên `.detach()`, đồ thị tính toán sẽ lan truyền gradient ngược về Generator ngay trong bước cập nhật của Discriminator, làm hỏng toàn bộ quá trình học đối kháng (Adversarial Training).
>
> 2. **Quy ước đặt tên biến (Reflection Matching) của `BaseModel`**:
>    * Phần tử trong `self.model_names = ['G_A']` $\iff$ Biến mạng phải là `self.netG_A`.
>    * Phần tử trong `self.loss_names = ['cycle_A']` $\iff$ Biến lưu giá trị loss phải là `self.loss_cycle_A`.
>    * Phần tử trong `self.visual_names = ['fake_B']` $\iff$ Biến lưu ảnh phải là `self.fake_B`.
>    *Nếu đặt sai tiền tố, các hàm tự động của `BaseModel` (lưu model, log loss, vẽ ảnh) sẽ bị lỗi AttributeError.*
>
> 3. **Kích thước ảnh đầu vào**:
>    Kiến trúc `ResnetGenerator` mặc định downsample 2 lần ($2^2 = 4$), trong khi `UnetGenerator` downsample 7 hoặc 8 lần ($2^7 = 128$ hoặc $2^8 = 256$). Chiều cao và chiều rộng của ảnh đầu vào **bắt buộc phải chia hết cho hệ số downsample tương ứng** để tránh lỗi mismatch kích thước tensor ở các tầng skip connection và upsampling.
