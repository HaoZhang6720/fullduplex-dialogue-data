"""Topic pools used to seed multi-round dialogues.

Topics are kept in two parallel pools (English and Chinese). They cover daily
life, productivity, entertainment, technology, lifestyle, public-interest, and
more. Edit or extend these lists to bias the generated corpus toward your own
domain.
"""

EN_TOPICS = [
    "Weather forecast", "Restaurant recommendations", "Movie suggestions", "Music recommendations",
    "Local news", "International news", "Traffic updates", "Public transport schedules", "Airline flight status",
    "Hotel bookings", "Taxi service", "Directions to a location", "Nearby gas stations", "Tourist attractions",
    "Current time", "Date and time", "Currency exchange rates", "Stock market updates", "Sports scores",
    "Event reminders", "Calendar scheduling", "Setting alarms", "Setting timers", "Fitness tracking",
    "Daily step count", "Calories burned", "Sleep tracking", "Heart rate monitoring", "Upcoming meetings",
    "Weather alerts", "Health tips", "Nutrition advice", "Cooking recipes", "Workout suggestions",
    "Book recommendations", "Podcast recommendations", "TV show suggestions", "Online shopping deals",
    "Order tracking", "Customer service queries", "Tech support", "App troubleshooting", "System updates",
    "Gadget recommendations", "Language translation", "Currency conversion", "Unit conversion",
    "Math problem solving", "Spelling assistance", "Grammar correction", "Word definitions",
    "Synonyms and antonyms", "Trivia questions", "History facts", "Geography trivia", "Space exploration",
    "Environmental issues", "Climate change", "Recycling tips", "Sustainability advice", "Animal facts",
    "Bird watching tips", "Wildlife conservation", "Home maintenance tips", "Gardening advice",
    "Indoor plant care", "Pet care tips", "Dog training tips", "Cat behavior", "Fish tank maintenance",
    "Bird feeding tips", "Home cleaning hacks", "Home organization tips", "Minimalism advice",
    "Meditation guidance", "Breathing exercises", "Yoga tips", "Stress management", "Mental health resources",
    "Counseling services", "Legal advice", "Job search tips", "Resume building", "Interview preparation",
    "Salary negotiation", "Workplace productivity", "Career development", "Leadership advice",
    "Team management", "Conflict resolution", "Public speaking tips", "Presentation skills", "Time management",
    "Study techniques", "Learning new skills", "Online courses", "Education resources", "Language learning tips",
    "College applications", "Scholarship information", "Student loans", "Exam preparation",
    "Test-taking strategies", "Virtual tours", "Online museums", "Art history facts", "Art techniques",
    "Photography tips", "Video editing advice", "Graphic design tips", "Web development", "App development",
    "Programming languages", "Code debugging", "Software development", "Cybersecurity advice", "Data privacy tips",
    "Cloud storage", "Online collaboration tools", "Social media marketing", "Email marketing",
    "Digital advertising", "Brand building", "Content creation", "Blog writing tips", "SEO strategies",
    "E-commerce platforms", "Small business tips", "Freelancing advice", "Remote work tips", "Home office setup",
    "Healthy work-life balance", "Parental advice", "Child development", "Parenting challenges", "School choices",
    "Homework help", "After-school activities", "Family vacation planning", "Road trip ideas", "Camping tips",
    "Hiking trails", "National parks", "Wildlife spotting", "Beach vacations", "Travel safety tips",
    "Packing advice", "Travel insurance", "Travel deals", "Local food recommendations", "Street food tips",
    "Fine dining options", "Vegetarian recipes", "Vegan diet tips", "Gluten-free recipes", "Healthy snacks",
    "Baking tips", "Dessert recipes", "Coffee brewing tips", "Tea varieties", "Wine pairing tips",
    "Craft beer suggestions", "Cocktail recipes", "Juicing tips", "Smoothie recipes", "Mindfulness techniques",
    "Goal setting tips", "Motivation strategies", "Positive thinking", "Building good habits", "Breaking bad habits",
    "Self-care tips", "Hobby suggestions", "Board game recommendations", "Card game rules", "Sports rules",
    "Exercise equipment", "Running tips", "Cycling routes", "Swimming techniques", "Tennis tips", "Golf tips",
    "Soccer drills", "Basketball strategies", "Cricket techniques", "Baseball tips", "American football rules",
    "Martial arts techniques", "Dance routines", "Music theory", "Instrument learning tips", "Singing tips",
    "Songwriting advice", "Film production tips", "Stage performance tips", "Theater history", "Magic tricks",
    "Comedy routines", "Puzzle games", "Brain teasers", "Chess strategies", "Crossword puzzles", "Sudoku tips",
    "Word games", "Home renovation tips", "DIY projects", "Car maintenance tips", "Bicycle repair",
    "Motorcycle safety", "Electric vehicle information", "Fuel efficiency tips", "Public transportation updates",
    "Bike-sharing services", "Ride-sharing tips", "Smart home automation", "Home security systems",
    "Energy-saving tips", "Electricity bill reduction", "Water conservation", "Waste reduction strategies",
    "Composting advice", "Sustainable gardening", "Eco-friendly products", "Recycling center locations",
    "Second-hand shopping tips", "Thrift store recommendations", "Minimalist lifestyle advice", "Zero-waste living",
    "Digital detox tips", "Time blocking techniques", "Budgeting tips", "Personal finance management",
    "Saving for retirement", "Investment strategies", "Cryptocurrency trends", "Real estate market updates",
    "Mortgage advice", "Home buying tips", "Renting vs buying", "Property management", "House flipping tips",
    "Interior design inspiration", "Furniture arrangement tips", "Lighting design", "Decorating small spaces",
    "Open floor plans", "Modern architecture", "Mid-century modern design", "Industrial design", "Scandinavian design",
    "Traditional architecture", "Famous architects", "Green building design", "Smart city planning",
    "Historic buildings", "Cultural heritage conservation", "Architectural landmarks", "Sustainable architecture",
    "Urban planning strategies", "Public park design", "City infrastructure", "Noise pollution reduction",
    "Traffic management", "Public space utilization", "Community building tips", "Volunteering opportunities",
    "Fundraising tips", "Nonprofit organization management", "Social justice initiatives", "Diversity and inclusion",
    "Human rights issues", "Political activism", "Civic engagement", "Public policy analysis",
    "Law enforcement practices", "Judicial system overview", "Legal rights", "Intellectual property law",
    "Copyright regulations", "Business law basics", "Consumer protection laws", "Privacy laws",
    "Environmental regulations", "Sustainability in business", "Corporate social responsibility",
    "Climate change mitigation", "Renewable energy options", "Solar power", "Wind energy", "Electric vehicles",
    "Battery technology", "Nuclear energy", "Fossil fuels", "Natural disaster preparedness",
    "Emergency response tips", "First aid techniques", "CPR instructions", "Fire safety", "Home safety",
    "Cyberbullying prevention", "Digital citizenship", "Online privacy protection", "Internet safety for kids",
    "Social media privacy", "Online reputation management", "Digital footprints", "Password management",
    "Cybersecurity best practices", "Phishing scams", "Identity theft prevention", "Online shopping safety",
    "App security", "Mobile device safety", "Artificial intelligence developments", "Machine learning applications",
    "Robotics in daily life", "Self-driving cars", "Space travel", "Smartphone trends", "Wearable technology",
    "Fitness trackers", "Augmented reality applications", "Virtual reality games", "3D printing technology",
    "Blockchain technology", "Quantum computing", "Genetics and DNA analysis", "Biomedical research",
    "Healthcare innovations", "Telemedicine", "Mental health apps", "Meditation apps", "Fitness apps",
    "Diet tracking apps", "Sleep analysis", "Wearable health devices", "Vaccination information",
    "Healthy lifestyle choices", "Preventive healthcare tips", "Mental wellness", "Workplace wellness programs",
    "Workplace ergonomics", "Burnout prevention", "Job satisfaction tips", "Healthy work culture",
    "Remote team collaboration", "Productivity apps", "Task management tools", "Project management software",
    "Video conferencing platforms", "Collaborative whiteboards", "Presentation tools", "Document sharing platforms",
]

ZH_TOPICS = [
    "天气预报", "餐厅推荐", "电影推荐", "音乐推荐", "本地新闻", "国际新闻", "交通更新", "公共交通时刻表", "航空航班状态",
    "酒店预订", "出租车服务", "地点导航", "附近的加油站", "旅游景点", "当前时间", "日期和时间", "货币汇率", "股市更新",
    "体育比分", "活动提醒", "日历安排", "设定闹钟", "设定计时器", "健身跟踪", "每日步数", "燃烧的卡路里", "睡眠跟踪",
    "心率监测", "即将举行的会议", "天气警报", "健康提示", "营养建议", "烹饪食谱", "锻炼建议", "书籍推荐", "播客推荐",
    "电视节目推荐", "在线购物优惠", "订单跟踪", "客户服务查询", "技术支持", "应用故障排除", "系统更新", "小工具推荐",
    "语言翻译", "货币转换", "单位转换", "数学问题求解", "拼写帮助", "语法纠正", "单词定义", "同义词和反义词", "小知识问答",
    "历史事实", "地理小知识", "太空探索", "环境问题", "气候变化", "回收建议", "可持续性建议", "动物知识", "观鸟技巧",
    "野生动物保护", "家庭维护建议", "园艺建议", "室内植物护理", "宠物护理技巧", "狗训练技巧", "猫的行为", "鱼缸维护",
    "喂鸟技巧", "家居清洁窍门", "家庭整理技巧", "极简主义建议", "冥想指导", "呼吸练习", "瑜伽技巧", "压力管理",
    "心理健康资源", "心理咨询服务", "法律建议", "求职技巧", "简历编写", "面试准备", "薪资谈判", "工作效率", "职业发展",
    "领导力建议", "团队管理", "冲突解决", "公共演讲技巧", "演示技巧", "时间管理", "学习技巧", "学习新技能", "在线课程",
    "教育资源", "语言学习技巧", "大学申请", "奖学金信息", "学生贷款", "考试准备", "考试策略", "虚拟旅游", "在线博物馆",
    "艺术史知识", "艺术技巧", "摄影技巧", "视频编辑建议", "平面设计技巧", "网页开发", "应用开发", "编程语言", "代码调试",
    "软件开发", "网络安全建议", "数据隐私提示", "云存储", "在线协作工具", "社交媒体营销", "电子邮件营销", "数字广告",
    "品牌建设", "内容创作", "博客写作技巧", "SEO策略", "电商平台", "小型企业建议", "自由职业建议", "远程工作技巧",
    "家庭办公设置", "健康的工作与生活平衡", "育儿建议", "儿童发展", "育儿挑战", "学校选择", "作业帮助", "课外活动",
    "家庭度假规划", "公路旅行想法", "露营技巧", "徒步路线", "国家公园", "野生动物观赏", "海滩度假", "旅行安全提示",
    "打包建议", "旅游保险", "旅游优惠", "本地美食推荐", "街头小吃技巧", "高档餐饮选择", "素食食谱", "素食主义饮食建议",
    "无麸质食谱", "健康零食", "烘焙技巧", "甜点食谱", "咖啡冲泡技巧", "茶种类", "葡萄酒搭配建议", "精酿啤酒推荐",
    "鸡尾酒食谱", "果汁技巧", "奶昔食谱", "正念技巧", "目标设定技巧", "激励策略", "积极思维", "建立良好习惯", "打破坏习惯",
    "自我护理技巧", "兴趣爱好建议", "桌游推荐", "卡牌游戏规则", "体育规则", "健身器材", "跑步技巧", "骑行路线",
    "游泳技巧", "网球技巧", "高尔夫技巧", "足球训练", "篮球策略", "板球技巧", "棒球技巧", "美式足球规则", "武术技巧",
    "舞蹈动作", "音乐理论", "乐器学习技巧", "唱歌技巧", "歌曲创作建议", "电影制作技巧", "舞台表演技巧", "戏剧历史",
    "魔术技巧", "喜剧节目", "益智游戏", "脑筋急转弯", "国际象棋策略", "填字游戏", "数独技巧", "文字游戏", "家庭装修建议",
    "DIY项目", "汽车维护技巧", "自行车修理", "摩托车安全", "电动车信息", "燃油效率技巧", "公共交通更新", "自行车共享服务",
    "拼车技巧", "智能家居自动化", "家庭安全系统", "节能技巧", "电费减少", "节水", "减少浪费策略", "堆肥建议",
    "可持续园艺", "环保产品", "回收中心位置", "二手购物技巧", "二手店推荐", "极简主义生活建议", "零浪费生活",
    "数字排毒技巧", "时间管理技巧", "预算建议", "个人理财管理", "退休储蓄", "投资策略", "加密货币趋势", "房地产市场更新",
    "抵押贷款建议", "购房技巧", "租房与购房对比", "物业管理", "房屋翻新技巧", "室内设计灵感", "家具摆放技巧", "照明设计",
    "小空间装饰", "开放式布局", "现代建筑", "中世纪现代设计", "工业设计", "北欧设计", "传统建筑", "著名建筑师",
    "绿色建筑设计", "智慧城市规划", "历史建筑", "文化遗产保护", "建筑地标", "可持续建筑", "城市规划策略", "公共公园设计",
    "城市基础设施", "减少噪音污染", "交通管理", "公共空间利用", "社区建设建议", "志愿服务机会", "筹款技巧",
    "非营利组织管理", "社会公正倡议", "多样性与包容性", "人权问题", "政治活动", "公民参与", "公共政策分析", "执法实践",
    "司法系统概述", "法律权利", "知识产权法", "版权法规", "商业法基础", "消费者保护法", "隐私法", "环境法规",
    "企业可持续发展", "企业社会责任", "气候变化缓解", "可再生能源选项", "太阳能", "风能", "电动汽车", "电池技术",
    "核能", "化石燃料", "自然灾害准备", "紧急响应技巧", "急救技术", "心肺复苏术指导", "消防安全", "家庭安全",
    "网络欺凌预防", "数字公民", "在线隐私保护", "儿童互联网安全", "社交媒体隐私", "在线声誉管理", "数字足迹",
    "密码管理", "网络安全最佳实践", "网络钓鱼诈骗", "身份盗窃预防", "网上购物安全", "应用程序安全", "移动设备安全",
    "人工智能发展", "机器学习应用", "机器人在日常生活中的应用", "自动驾驶汽车", "太空旅行", "智能手机趋势",
    "可穿戴技术", "健身追踪器", "增强现实应用", "虚拟现实游戏", "3D打印技术", "区块链技术", "量子计算", "基因和DNA分析",
    "生物医学研究", "医疗创新", "远程医疗", "心理健康应用", "冥想应用", "健身应用", "饮食跟踪应用", "睡眠分析",
    "可穿戴健康设备", "疫苗信息", "健康生活方式选择", "预防性健康提示", "心理健康", "职场健康计划", "职场人体工学",
    "预防倦怠", "工作满意度建议", "健康的工作文化", "远程团队协作", "生产力应用", "任务管理工具", "项目管理软件",
    "视频会议平台", "协作白板", "演示工具", "文档共享平台",
]


def get_topics(language: str = "en"):
    """Return the topic pool for the requested language.

    Args:
        language: ``"en"`` for English, ``"zh"`` for Chinese.

    Raises:
        ValueError: when ``language`` is not recognized.
    """
    lang = language.lower()
    if lang in ("en", "english"):
        return EN_TOPICS
    if lang in ("zh", "chinese", "cn", "zh-cn"):
        return ZH_TOPICS
    raise ValueError(f"Unknown language: {language!r}. Use 'en' or 'zh'.")
