# magents v0.1

最小主 Agent → 执行 Agent → 核查 Agent → 可选修正 → 最终回答。
角色各自连续上下文；运行器维护轻量任务状态。运行器自动保存完整返回，展示截断部分可按事件 ID 展开。

## 运行

在本目录运行，推荐使用已有 earth-agent Python：

```bash
/srv/user/yangxiao/anaconda3/envs/earth-agent/bin/python manage.py preflight
/srv/user/yangxiao/anaconda3/envs/earth-agent/bin/python manage.py demo --output var/runs/demo-01
/srv/user/yangxiao/anaconda3/envs/earth-agent/bin/python manage.py run --task /path/public-task.txt --data /path/public-data-only --output var/runs/task-01
python manage.py status --output var/runs/task-01
PYTHONPATH=src:../python-agent/src python -m unittest discover -s tests -v
```

`run` 会真实调用 API，无二次确认；`demo` 和 `preflight` 不调用 API。
输出目录必须是新的，避免覆盖旧实验。当前中断后保留状态和日志，但不支持自动恢复。

## 配置和授权

`configs/default.json` 引用旧 API 配置及 credentials 文件，不复制密钥。默认 Qwen，`--profile` 可选择已有 DeepSeek profile。
默认每个任务共享最多 40 次 API 请求，每次输出最多 4096 token；`max_role_requests: 12` 是单次调度片段长度，不再是执行阶段的终止上限。片段耗尽后自动交回部分证据，由主 Agent 选择 CONTINUE 或 STOP。继续使用同一角色上下文和工作文件，不重新创建执行 Agent。
`downstream_reserve: 6` 为执行后的核查及收尾预留请求；核查至少留一次请求用于最终回答。额度不足时保留部分结果，最终状态为 partial，不冒充已完成计算。默认最多一轮修正。
同一模型回复可以提交多个互不依赖的工具调用；框架不规定具体工具、批次大小或题目计算方法。
执行阶段最后一个可用请求预留给 `submit_progress`：累计已完成、待完成、问题、产物、证据 ID 和下一步。这占用既有调度片段和共享预算，不额外加请求。也可以提前提交。
主 Agent 看到原计划与最新累计进度，只安排剩余操作；重算需说明理由。角色上下文和工具 schema 保持连续，交接提醒追加到末尾。
核查阶段用 `submit_review` 提交 checked/issues/unchecked 和证据，不默认重新解题。最后一请求同样预留提交。
审查提示词（2026-10-30）要求将执行摘要与原因解释视为未验证说法：优先展开关键原始调用，
核对实际参数类型、接口和完整返回，寻找反证；必要时在既有权限内做最小区分检查。
未独立核实的说法必须列入 unchecked，不得仅凭执行者断言认定工具、环境或数据受限。
这是模型行为要求，不是运行器保证每项结论都经过验证，也不会新增代码执行权限。
最终用 `submit_final` 提交 completed 答案或 incomplete 原因。普通文本、伪工具标签和截断调用均不算提交；没有提交时 answer 保持空，不能算完成。
提交检查仅验证结构、引用存在及明显字段矛盾，不证明内容正确。缺交接时保存 handoff_missing，不把最后一句话伪装成总结。
不增加产物文件：进度和正式最终提交写入 state.json，全部交接/核查/提交记录进入 events.jsonl。
这些是防止失控的技术限制，不是人民币预算保证。记录供应商返回的原始 usage（包括缓存字段），不把缺失用量当成零成本；当前不计算人民币费用或跨运行累计额度。
后续明确要求执行的实验，在已约定范围内直接运行，不逐题、逐请求重复确认。扩大实验范围或外部系统要求的权限仍单独处理。

## 工具与隔离

### 可选假设验证（2026-10-29）

EarthBench 入口增加 `--hypothesis-verification`，默认关闭，可与 `--official-only` 合用。
这是一种新增诊断权限的实验条件，不等同于历史严格禁止辅助代码的对照条件。
执行/审查 Agent 使用 `hypothesis_propose` 登记假设、替代解释、检查计划和支持/反对预期；
用 `hypothesis_test` 调用官方工具或 `run_command`，再用 `hypothesis_conclude` 引用同一假设下的检查 ID。
协调者仍只协调，不执行诊断。实现位于 `src/magents/hypotheses.py`，通过 adapters 和 engine 接入。
诊断命令使用官方工具 runner 的挂载和 Python 环境，但处于独立进程，工作目录为 workspace，
不共享桥接进程的内存或当前目录。提示要求只做诊断、不得替代最终官方计算；
任意命令的语义用途并非技术强制隔离，仍须审查轨迹。仅检查引用归属，不自动认证结论。
登记、命令和结论进入既有 events.jsonl，不新增逐假设文件，也不增加自动付费请求或隐藏重试。
验证前可列出 `required_inputs`，模块在同一官方沙箱检查文件存在且可读；缺失则不调用目标工具，
记录为未执行，不允许仅凭这些记录给出支持/否定结论。空列表适用于无需文件或验证缺失本身，
不会自动判断 Agent 是否漏报输入；文件存在检查也不证明栅格内容和配对正确。
诊断预检是一次本地命令，不新增模型请求；不自动转换参数或复制旧运行产物。

复用相邻 `python-agent/src/rsagent` 的 API 客户端、NamespaceRunner 和 EarthToolProvider；只依赖正式运行包，不依赖实验代码。
默认先关闭官方工具桥（`official_tools: false`）；需要 EarthBench 时改为 true，读取已有 API 配置中的 earth_tools。
命令只能在 namespace 沙箱执行，不回退到无隔离本地命令。沙箱不可用会返回错误。
输入只挂载明确指定的数据目录，不继承旧配置中宽泛的 read_only_paths。
v0.1 拒绝数据目录中的 JSON、脚本和符号链接，防止直接挂载包含 GT 的题目目录。此检查不是语义脱敏器；操作者仍需保证 CSV 等输入不包含答案。
图片目前仅能作为文件处理，不会送入视觉模型。
核查 Agent 可以独立运行代码，使用 review 前缀文件是提示约定，并非只读权限隔离。

## 产物与边界

每次运行只建立 run.json、state.json、events.jsonl、workspace/。
日志包含交接、模型响应、工具调用、原始返回和 usage。网络传输前的失败也计入请求上限，无自动重试。
当前是串行协调闭环，不是并发子任务调度。续执行指同一进程内调度，不代表支持退出后的恢复。核查 PASS 是意见，不是评分；completed 表示正式提交完成且没有调度侧未完成标记，不证明答案正确。
EarthBench 实验入口已支持宿主侧评分；Q10 学习试验支持轨迹对比和独立回放。Skill 检索已接入，Tool 发布和自动库更新尚未接入。
这是一份可以运行和扩展的执行原型，不是完整四阶段进化系统，也不宣称已复现旧 baseline。

## Skill 库与检索

正式库位于 `library/skills/`，当前一份模型提取的 Skill 经用户授权以
`provisional` 状态暂时加入。正文保留原候选，检索元数据及验证范围由维护者补充；
原题配对两组均正确，不能称为已证明增益或迁移有效。

默认 `skill_retrieval: true`。协调 Agent 在规划阶段调用 `skill_search(query)`，
检索名称、适用场景、关键词和工具依赖；最多返回五条摘要。随后通过
`skill_read(skill_id)` 阅读正文和边界，自主决定采用或拒绝。执行和核查角色也可
按需搜索。当前是确定性的关键词匹配，不是向量 RAG，没有额外检索模型；不保证
同义词召回或强制模型一定执行搜索。所有调用记录在 events.jsonl，库在启动时
加载为内存快照并记录 skill_library_snapshot，运行中不会跟随库变化。

协调角色只新增只读检索权限，不能执行计算。Skill 库不挂载到代码沙箱。
禁用辅助代码不禁用 Skill 检索，两者是独立实验因素。

```bash
# 检索模式：只通过工具按需阅读，不预先注入全文
python experiments/earthbench_q1.py --task-id 93 --official-only --output var/runs/q93-retrieval-01
# 无 Skill 对照
python experiments/earthbench_q1.py --task-id 93 --official-only --no-skill-retrieval --output var/runs/q93-noskill-01
```

`manage.py run` 同样支持 `--no-skill-retrieval`。`--skill-file` 的直接注入实验
自动关闭库检索，避免混用；现有 paired_controls 和学习回放也显式关闭库检索。
上述命令为使用示例，不代表已执行新的付费评测。

### 规划与检索的交接

有/无 Skill 两组都用 `submit_plan(plan)` 显式交接，不再靠没有工具调用的
普通文字隐式结束规划。规划提示明确区分“执行角色可用能力”与“协调角色可调用
的检索工具”；协调角色不得查看输入文件或直接计算。

`max_plan_requests: 4` 包含检索及提交，计入原有共享预算，不增加总请求数。
正常可搜索、阅读后提前提交；最后一个规划请求只开放 submit_plan，并停止检索。
只有这个阶段收尾边界缩减工具 schema，其余上下文保持追加；该边界可能影响
请求缓存前缀，不能声称所有请求 schema 完全相同。

若规划仍超限或没有有效提交，记录 `plan_status: fallback` 和 planning_fallback
事件，交由执行 Agent 根据原始任务自主规划。成功检索的证据编号随交接传递；
不把未提交的文字当成正式计划，也不把规划失败伪装成 Skill 验证成功。
API/网络异常仍按原逻辑报告，不作为可忽略的规划超时。旧实验日志不改写，
修复后需重新配对运行才能评价收益。
