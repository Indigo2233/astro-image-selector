# AstroImageSelector

Windows x64 便携版从 [GitHub Releases](https://github.com/Indigo2233/astro-image-selector/releases) 下载。选择包含 `.fit` 或 `.fits` 文件的目录，再开始分析。图表和文件表都可以打开预览。按阈值标记文件后，可将文件移动到项目的 `rejected` 子目录。

AstroImageSelector 是使用 Rust 和 eframe/egui 开发的 Windows 桌面应用，用于 FITS 图像质量分析、筛选和预览。应用启动时会检测显示适配器。检测到独立显卡时默认开启 GPU 快速模式；仅有集成显卡或没有兼容的硬件计算设备时默认使用 CPU。GPU 快速模式使用受支持的显卡计算 FWHM 和椭率。GPU 初始化失败、运行失败或没有有效测量时，应用自动使用 CPU 精确计算。可在分析控制中关闭 GPU 快速模式。

分析线程数会按物理核心数和可用内存自动填写。点击线程数旁的“自动”可重新计算推荐值。N100 的推荐值为 3 个线程。

分析完成后，结果写入项目根目录的 `.astro_selector_analysis_cache.json`。再次分析同一目录时，文件路径、大小和修改时间均未变化的图像直接读取缓存，只重新计算新增或已修改的图像。目录移动后仍可复用缓存。损坏或版本不兼容的缓存会被忽略并重建。

分析目录中的平场素材会在计算和筛选前自动排除。识别依据包括 `IMAGETYP`、`IMAGETYPE`、`FRAME`、`OBSTYPE`、`OBSTYP` 等常见 FITS 头字段，以及文件名中的 `flat` 或 `Flat`。状态栏会显示本次排除的平场数量。

## 构建和验证

从项目根目录运行以下命令。

```powershell
$env:PATH = "C:\msys64\ucrt64\bin;$env:PATH"
$env:RUSTFLAGS = "-C target-feature=+crt-static"
cargo build --workspace --release --locked --target x86_64-pc-windows-gnu
cargo test --workspace --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
```

发行版使用 MinGW-w64 UCRT 工具链。最终 EXE 只依赖 Windows 系统组件。第三方算法声明见 [Photutils 和 SciPy 许可证](licenses/PHOTUTILS-AND-SCIPY.txt)。

版本标签会触发 GitHub Actions 构建 Windows x64 portable EXE，并将可执行文件附加到 Release。仓库源码不保存发行二进制。

在 50 颗拟合星点的上限恰好跨过同峰值组时，Rust 按输入顺序从后向前选择并列星点。NumPy 的选取次序随 CPU 排序后端变化。移动文件失败时，界面保留失败的文件和标记，可再次尝试。
