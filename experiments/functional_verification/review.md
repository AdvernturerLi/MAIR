# MAIR 功能验证核查表（12 例）

> 生成日期：2026-10-10。验证「MAIR-available-noexp-v1」在三阶段先验、两级智能体、
> 工具选择、反思与回退上的功能正确性。**"未崩溃"不等于"智能体判断完全正确"**——
> 本表把感知/规划的判断正确性单独评估。

## 基线信息

| 项 | 值 |
|---|---|
| 版本 | MAIR-available-noexp-v1 |
| Git commit | `a16a661`（代码）+ `e425a6d`（环境基线） |
| 规划/选工具模型 | deepseek-v4-flash-vision-exp（替代 GPT-4o） |
| temperature / max_tokens | 0.0 / 8192 |
| STOP_THRESHOLD | low（严格低于即停） |
| 经验库 | no-experience（`--no-retrieval` 等价） |
| GPU / torch | A100-40GB / 1.13.1+cu117 |
| DepictQA commit | 06583c6（Vicuna-7B + degra_eval.pt / DQ495K_Abstractor.pt） |
| 验证图来源 | MiO_test `original/GT/001.png` 合成，固定 seed 42 |

---

## 逐例记录

### s1_jpeg — success（完全正确）
- **人工标注退化**：jpeg compression artifact
- **DepictQA 识别**：jpeg=high，其余 very low
- **规划**：[jpeg compression artifact removal]
- **首选工具**：fbcnn_5
- **执行路径**：候选[fbcnn_5, fbcnn_blind, swinir_40] → fbcnn_5 → very low → 达阈值停
- **工具异常**：无　**回退**：无

### s2_noise — success（完全正确）
- **人工标注退化**：noise
- **DepictQA 识别**：noise=high
- **规划**：[denoising]　**首选**：swinir_50
- **执行路径**：候选[swinir_50, restormer, xrestormer] → swinir_50 → very low → 停
- **工具异常**：无　**回退**：无

### s3_rain — success（完全正确）
- **人工标注退化**：rain
- **DepictQA 识别**：rain=very high
- **规划**：[deraining]　**首选**：xrestormer
- **执行路径**：候选[xrestormer, restormer, mprnet] → restormer → very low → 停
- **工具异常**：无　**回退**：无

### s4_lowres — success（部分正确）
- **人工标注退化**：low resolution
- **DepictQA 识别**：dark=medium（**误检**）、motion blur=low（未进计划）；low resolution 由尺寸启发式触发
- **规划**：[super-resolution, brightening]（三阶段顺序正确，但 brightening 是多余任务）
- **首选**：SR=swinir_psnr，brightening=gamma_correction
- **执行路径**：SR 候选[swinir_psnr, xrestormer, hat] 全 low → 比较选 hat；brightening → constant_shift → very low
- **工具异常**：无　**回退**：无
- **备注**：低分辨率图被误判 dark，多做了一次 brightening（本不该有的操作）

### c1_noise_jpeg — success（完全正确）
- **人工标注退化**：noise + jpeg
- **DepictQA 识别**：noise=very high、jpeg=high
- **规划**：[jpeg, denoising]（compression→imaging 正确）
- **首选**：jpeg=fbcnn_5，denoising=swinir_15
- **执行路径**：fbcnn_5→very low；swinir_15→very low
- **工具异常**：无　**回退**：无

### c2_rain_jpeg — success（完全正确）
- **人工标注退化**：rain + jpeg
- **DepictQA 识别**：rain=medium、jpeg=medium
- **规划**：[jpeg, deraining]（compression→scene 正确）
- **首选**：jpeg=swinir_40，deraining=xrestormer
- **执行路径**：swinir_40→very low；restormer→very low
- **工具异常**：无　**回退**：无

### c3_haze_lowres — success（完全正确，修复后）
- **人工标注退化**：haze + low resolution
- **DepictQA 识别**：haze=very high；low resolution 由尺寸启发式触发
- **规划**：[super-resolution, dehazing]（imaging→scene 正确）
- **首选**：SR=swinir_psnr，dehazing=xrestormer
- **执行路径**：SR 候选[swinir_psnr, xrestormer] 均未达 very low → 比较选 swinir_psnr；dehazing xrestormer→very low
- **工具异常**：无　**回退**：无
- **备注**：修复前 SR 会被 blur 代理误判 very low 而跳过；修复后正确执行

### c4_rain_noise — success（完全正确）
- **人工标注退化**：rain + noise
- **DepictQA 识别**：rain=medium、noise=very high
- **规划**：[denoising, deraining]（imaging→scene 正确）
- **首选**：denoising=swinir_50，deraining=xrestormer
- **执行路径**：swinir_50→very low；xrestormer→very low
- **工具异常**：无　**回退**：无

### t1_noise_motionblur — success（部分正确）
- **人工标注退化**：motion blur + noise
- **DepictQA 识别**：noise=medium（✓）；motion blur=very low（**漏检**）；defocus blur=medium（**误检**：把 motion blur 当成 defocus blur）
- **规划**：[denoising]（defocus 无可用工具被跳过；motion blur 因漏检未进计划）
- **首选**：restormer
- **执行路径**：候选[restormer, mprnet, xrestormer] → mprnet → very low
- **工具异常**：无　**回退**：无
- **备注**：motion blur 被漏检且误判为 defocus blur，最终**未处理** motion blur

### t2_rain_haze — fallback_success（完全正确，经回退）
- **人工标注退化**：haze + rain
- **DepictQA 识别**：haze=high、rain=medium（均正确）
- **规划**：[dehazing, deraining]（**LLM 阶段内排序格式校验失败 5 次 → 回退严重度降序**）
- **首选**：dehazing=xrestormer，deraining=xrestormer
- **执行路径**：xrestormer→very low（雾）；xrestormer→very low（雨）
- **工具异常**：无　**回退**：是（LLM 排序回退到严重度降序）

### x1_rain_noise_jpeg — success（完全正确，三阶段完整）
- **人工标注退化**：rain + noise + jpeg
- **DepictQA 识别**：rain=high、noise=very high、jpeg=very high（全检出）
- **规划**：[jpeg, denoising, deraining]（compression→imaging→scene 完整正确）
- **首选**：jpeg=fbcnn_5，deraining=xrestormer
- **执行路径**：fbcnn_5→very low；denoising 输入已 very low → 跳过；deraining 三候选全 medium → 比较选 mprnet
- **工具异常**：无　**回退**：无

### x2_motionblur_noise_lowres — fallback_success（部分正确）
- **人工标注退化**：motion blur + noise + low resolution
- **DepictQA 识别**：noise=medium（✓）；motion blur=very low（**漏检**）；low resolution 由尺寸启发式触发
- **规划**：[denoising, super-resolution]（motion blur 因漏检未进计划）
- **首选**：denoising=restormer，SR=diffbir
- **执行路径**：denoising restormer→very low；SR 候选[diffbir, swinir_gan, hat]：**diffbir CUDA OOM**（DepictQA 驻留 28G + diffbir 生成式扩散显存不足）→ 记录错误并继续 → swinir_gan(low)/hat(medium) 比较 → 选 swinir_gan
- **工具异常**：是（diffbir CUDA OOM）　**回退**：是（diffbir → swinir_gan）
- **备注**：motion blur 漏检（未处理）；diffbir OOM 回退成功

---

## 统计

| 指标 | 定义 | 值 |
|---|---|---|
| 系统运行成功率 | 流程跑完不崩溃 | **12/12 = 100%** |
| 最终任务完成率 | 人工标注的退化全部被处理 | **10/12 = 83.3%**（t1、x2 遗漏 motion blur） |
| 完全感知正确率 | 感知无遗漏且无误检 | **9/12 = 75%** |
| 部分正确率 | 感知有遗漏或误检但仍跑完 | **3/12 = 25%**（s4 误检 dark、t1 漏检+误判 motion、x2 漏检 motion） |
| 规划正确率 | 三阶段顺序（compression→imaging→scene）正确 | **12/12 = 100%** |
| 工具回退率 | 工具执行失败后回退其他候选 | **1/12 = 8.3%**（x2：diffbir OOM→swinir_gan） |

另：**LLM 排序回退 1 次**（t2：格式校验失败→严重度降序）；平均单图耗时约 48s（中位 ~31s，x2 因 diffbir OOM 拖到 224.7s）。

## 关键发现

1. **三阶段先验 100% 正确**：所有计划严格遵循 compression→imaging→scene，跨阶段/同阶段/三退化均如此。
2. **motion blur 是 DepictQA 的薄弱项**：t1、x2 两例 motion blur 均被漏检（very low），t1 还被误判成 defocus blur。这是感知模型固有局限，非代码 bug。
3. **dark 误检**：低分辨率图（s4）被误判 dark=medium，多做一次 brightening。
4. **diffbir 生成式超分在 DepictQA 驻留时 OOM**（40G 不够同时容纳 2×DepictQA 14G + diffbir 扩散模型），但错误处理回退成功。
5. **LLM（DeepSeek）阶段内排序偶发格式失败**，回退严重度降序可保证流程不中断。
6. 修复后：SR 不再被 blur 代理误跳过（c3）、defocus 无工具时正确跳过不崩溃（t1）。

## 附录

- **轨迹目录**：`experiments/functional_verification/runs/{case_id}/trajectory.json`（12 份）
- **汇总**：`experiments/functional_verification/summary.json`
- **合成脚本**：`experiments/make_verification_set.py`；**批跑脚本**：`experiments/run_verification.py`
- **配置**：`config.yml`（API key 已脱敏，不入库）；环境基线见 `experiments/environment/environment.md`
- **本轮修复的代码**：Restormer `required_weight_files`、SR 低分辨率不跳过、LLM 排序严重度降序回退、`setuptools==65.5.1`（diffbir 补 pkg_resources）
