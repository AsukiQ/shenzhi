# 论文检索与知识图谱 API 使用手册

## 基本信息

服务地址：`http://47.110.47.12`

接口统一返回 JSON。POST 请求需设置请求头：`Content-Type: application/json`；登录态基于 Cookie，浏览器请求需携带 `credentials: "include"`。

## 接口总览

|**业务需求**|**方法**|**路径**|**功能说明**|**请求参数与示例**|**返回示例**|
|---|---|---|---|---|---|
|**论文检索**<br>用户输入关键词、年份、会议等条件查找相关论文；或沿作者、主题等关系链路扩展查找；已知标题时按标题快速定位。|POST|`/api/retrieval/search`|论文增强检索（主入口）：按标题、摘要、元数据检索，融合 BM25、模糊召回与 Zilliz 向量召回，RRF 排序；支持中英文、模糊匹配与条件过滤。也可用 GET：`/api/retrieval/search?q=...&top_k=10`|**参数：**<br>query \(string, 必填\) 查询内容<br>top\_k \(int, 默认10\) 返回数量，建议≤20<br>year\_gte / year\_lte \(int\) 年份范围<br>conference / author / keyword / subject \(string\[\]\) 过滤列表<br>**请求体：**<br>`{`<br>`"query": "2022年以后图神经网络",`<br>`"top_k": 10, "year_gte": 2022,`<br>`"year_lte": 2026, "conference": ["AAAI"],`<br>`"author": [], "keyword": [], "subject": []`<br>`}`<br>按基金查找时将基金名写入 query，无独立 funding 参数。|`{`<br>`"results": [{`<br>`"paper_id": "paper:2605.24041v2",`<br>`"title": "...", "abstract": "...",`<br>`"conference": "ICML", "year": 2026,`<br>`"authors": [...], "keywords": [...],`<br>`"subjects": [...], "funding": [...],`<br>`"image_id": null, "score": 0.022,`<br>`"rank": 1, "source_scores": {},`<br>`"retrieval_mode": "bm25+fuzzy+zilliz_dense"`<br>`}],`<br>`"state": {}, "query_parse": {},`<br>`"query_rewrite": {}`<br>`}`<br>结果按 rank 排序，勿把 score 当百分比。subjects 为图谱主题；funding 为基金列表；authors 可能由图谱回填、顺序非署名序；image\_id 为文件名（非完整 URL）。|
||POST|`/api/retrieval/multistep-search`|多步关系检索：含作者、主题等关系约束的复杂查询。普通检索优先用 search。|**请求体：**<br>`{`<br>`"query": "同一作者在相同主题下的其他论文",`<br>`"top_k": 10, "max_steps": 6`<br>`}`|结构同 search，按关系多步扩展召回。|
||GET|`/api/kg/search?q={关键词}&limit=20`|按论文标题快速定位。需检索标题和摘要时用 /api/retrieval/search。|**参数：**<br>q \(string\) 标题关键词<br>limit \(int\) 返回数量 1–50，默认 20|匹配的论文列表。|
|**学者检索**<br>按姓名查找学者并获取学者画像。|GET|`/api/retrieval/scholars/search?q={姓名}&limit=20&offset=0`|按姓名或 author\_id 片段模糊检索学者，返回稳定的 scholar\_id 和论文数。|**参数：**<br>q \(string\) 姓名或 author\_id 片段<br>limit \(int, 默认20, 最大100\)<br>offset \(int, 默认0\)<br>**示例：**<br>`/api/retrieval/scholars/search?q=Marcel%20Binz&limit=20`|`{`<br>`"results": [{"scholar_id":"author:...", "name":"Marcel Binz", "paper_count":9}],`<br>`"query":"Marcel Binz"`<br>`}`<br>使用返回的 scholar\_id 获取详情。|
||GET|`/api/retrieval/scholars/{scholar_id}`|获取学者画像。|scholar\_id 必须使用搜索结果返回的稳定 ID，并进行 URL 编码；不要直接用姓名作为 ID。|`{`<br>`"scholar_id":"author:...", "name":"Marcel Binz",`<br>`"paper_count":9, "years":[2022,2023,2024],`<br>`"conferences":[...], "topics":[...], "funding":[],`<br>`"institutions":[], "coauthors":[...], "papers":[...]`<br>`}`|
|**主题检索**<br>按主题检索论文|GET|`/api/retrieval/search/by-subject?subject={主题}&top_k=10`|按主题检索论文，返回论文结果列表。|**参数：**<br>subject \(string, 必填\) 主题片段<br>top\_k \(int, 默认10\) 返回前 top\_k 篇相关论文<br>**示例：**<br>`/api/retrieval/search/by-subject?subject=reinforcement&top_k=10`|结构同论文检索，返回 results、state 等字段；results 中每项为论文。|
|**基金检索**<br>按基金检索论文|GET|`/api/retrieval/search/by-funding?funding={基金}&top_k=10`|按基金检索论文，返回论文结果列表。|**参数：**<br>funding \(string, 必填\) 基金名称片段<br>top\_k \(int, 默认10\) 返回前 top\_k 篇相关论文<br>**示例：**<br>`/api/retrieval/search/by-funding?funding=NSF&top_k=10`|结构同论文检索，返回 results、state 等字段；results 中每项为论文。|
|**基金候选检索**<br>按关键词查找基金|GET|`/api/retrieval/fundings/search`|按基金名称或 ID 片段匹配，不区分大小写；返回基金列表，按关联论文数降序排列。中文仅匹配已有中文文本，不自动翻译。按基金查论文用 /api/retrieval/search/by-funding（文本相关性检索，非 funding_id 精确过滤）。|**参数：**<br>q \(string, 可选\) 关键词，为空时浏览；中文或特殊字符需 URL 编码<br>limit \(int, 默认20, 1–100\)<br>offset \(int, 默认0, ≥0\) 分页偏移<br>**示例：**<br>`/api/retrieval/fundings/search?q=NSF&limit=20&offset=0`|`{`<br>`"results": [{`<br>`"funding_id": "funding:...",`<br>`"name": "...", "paper_count": 1`<br>`}], "query": "NSF"`<br>`}`<br>示意结构；无匹配返回空 results，非法分页参数返回 HTTP 400。已隔离确认的模板污染并规范化列表；不明确的资助声明保留，未逐篇核实资助关系。名称可能为资助声明片段；paper\_count 为当前图谱关联的去重论文数。|
|**机构检索**<br>按机构检索论文|GET|`/api/retrieval/search/by-institution?institution={机构}&top_k=10`|按机构名称或机构 ID 检索论文，返回论文结果列表；机构条件通过图谱中的论文—作者—机构关系进行过滤。|**参数：**<br>institution \(string, 必填\) 机构名称或机构 ID 片段<br>top\_k \(int, 默认10\) 返回前 top\_k 篇相关论文<br>**示例：**<br>`/api/retrieval/search/by-institution?institution=Stanford&top_k=10`|结构同论文检索，返回 results、state 等字段；results 中每项为论文。当前图谱中约 30,423 篇论文有明确机构关系，约占 536,241 篇 Paper 的 5.7%，其余论文可能没有机构数据。|
|**机构候选检索**<br>按关键词查找机构|GET|`/api/retrieval/institutions/search?q={关键词}&limit=20&offset=0`|按机构名称或机构 ID 片段检索机构，不区分大小写，返回机构候选列表及当前图谱中的关联论文数。按论文数降序排列；机构名称变体尚未合并，源数据可能含占位名称。|**参数：**<br>q \(string, 可选\) 机构名称或 ID 片段；为空时分页浏览机构<br>limit \(int, 默认20, 最大100\)<br>offset \(int, 默认0\) 分页偏移<br>**示例：**<br>`/api/retrieval/institutions/search?q=Stanford&limit=1`|`{"results":[{"institution_id":"institution:stanford_university","name":"Stanford University","paper_count":116}],"query":"Stanford"}`<br>paper\_count 为通过作者关系关联到的去重论文数，不代表机构全部发表成果；无匹配时 results 为 []。|
|**论文详情**<br>用户点击某篇论文，查看完整元数据、摘要、DOI 与 PDF 链接。|GET|`/api/kg/paper?paperId={paper_id}`|按 paperId 获取论文详情。paperId 从检索结果取得，调用需 URL 编码。|**路径参数：**<br>paperId \(string, 必填\)<br>**示例：**<br>`fetch("/api/kg/paper?paperId=" +`<br>`encodeURIComponent(id))`|`{`<br>`"paper_id": "paper:17203_...",`<br>`"title": "...", "authors": [],`<br>`"year": 2021, "venue": "AAAI",`<br>`"abstract": "...", "doi": "...",`<br>`"pdf_url": "..."`<br>`}`|
|**引用关系与知识图谱**<br>查看一篇论文引用了谁、被谁引用，以及关联的作者、机构、方法、基金、主题、会议等节点。|GET|`/api/kg/graph?paperId={paper_id}&depth=1`|获取论文引用关系与知识图谱。from=当前论文表示引用了 to；to=当前论文表示被 from 引用。|**参数：**<br>paperId \(string, 必填\) 论文 ID<br>depth \(int, 可选\) 展开深度 1–3，建议 1<br>|`{`<br>`"rootId": "当前论文ID",`<br>`"nodes": [],`<br>`"lines": [{`<br>`"from": "论文A", "to": "论文B",`<br>`"text": "CITES",`<br>`"data": {"type": "CITES"}`<br>`}]`<br>`}`|
|**论文库浏览与筛选**<br>按会议、年份、track 等条件浏览、分页查看论文列表；同时为筛选器下拉框提供会议、track、分类等元数据选项。|GET|`/api/papers`|论文列表，支持多条件筛选与分页。|**参数：**<br>q \(string\) 搜索关键词<br>conference \(string\) 会议名<br>year \(int\) 发表年份<br>track \(string\) track 名<br>limit \(int, 默认20\) 返回数量<br>offset \(int, 默认0\) 分页偏移<br>**示例：**<br>`/api/papers?q=graph&conference=AAAI&year=2024&limit=20&offset=0`|符合条件的论文列表。|
||GET|`/api/papers/venues`|获取所有出版物/会议列表（供筛选器）。|无参数。|出版物/会议列表。|
||GET|`/api/papers/tracks?conference=AAAI`|获取指定会议的 track 列表（供筛选器）。|conference \(string\) 会议名。|track 列表。|
||GET|`/api/categories`|获取论文分类列表（供筛选器）。|无参数。|分类列表。|
||GET|`/api/conferences`|获取所有会议列表（供筛选器）。|无参数。|会议列表。|
|**用户注册与登录**<br>新用户注册账号、登录获取会话、退出登录，以及在页面加载时识别当前登录用户。|POST|`/api/auth/register`|用户注册。|**请求体：**<br>`{"username":"...","password":"..."}`|注册结果。|
||POST|`/api/auth/login`|用户登录，成功后写 Cookie。|**请求体：**<br>`{"username":"username","password":"password"}`|登录态 Cookie。|
||POST|`/api/auth/logout`|退出登录。|无请求体。|清除登录态。|
||GET|`/api/auth/me`|获取当前登录用户信息。|需登录态 Cookie。|当前用户信息。|
|**收藏管理**<br>用户查看自己的收藏、添加感兴趣的论文/会议、以及取消收藏。|GET|`/api/favorites`|获取当前用户收藏列表。|需登录态 Cookie。|收藏列表。|
||POST|`/api/favorites`|添加收藏。|**请求体：**<br>`{"conferenceId":"conference-id"}`|添加结果。|
||DELETE|`/api/favorites/{conferenceId}`|取消指定收藏。|**路径参数：**<br>conferenceId \(string\) 要取消的收藏 ID。|删除结果。|
|**服务健康检查**<br>部署上线、监控巡检时确认主业务服务与检索服务是否正常、是否就绪。|GET|`/api/health`|检查主业务服务是否正常。|无参数。|服务健康状态。|
||GET|`/api/retrieval/health`|检查检索服务进程是否存活。|无参数。|进程健康状态。|
||GET|`/api/retrieval/ready`|检查检索服务是否加载完成、可对外服务。|无参数。|就绪状态。|

## 状态码

|**状态码**|**说明**|
|---|---|
|200|请求成功|
|201|创建成功|
|400|参数错误或缺少必填参数|
|401|未登录或认证失败|
|404|论文或资源不存在|
|413|请求内容过大|
|429|请求过于频繁|
|500|服务内部错误|
|503|服务暂时不可用|



> （注：部分内容可能由 AI 生成）
