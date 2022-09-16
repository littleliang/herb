import copy
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


def pc_cross(x, y, pc, cv):
	kf = KFold(n_splits=cv)  # 选定交叉验证方式
	rmse_cv = []
	for i in range(pc):
		rmse = []
		for train_index, test_index in kf.split(x):
			x_train, x_test = x[train_index], x[test_index]
			y_train, y_test = y[train_index], y[test_index]
			pls = PLSRegression(n_components=i + 1)
			pls.fit(x_train, y_train)
			y_predict = pls.predict(x_test)
			rmse.append(np.sqrt(mean_squared_error(y_test, y_predict)))
		rmse_mean = np.mean(rmse)
		rmse_cv.append(rmse_mean)
	index = np.argmin(rmse_cv)
	return rmse_cv, index


def cross(x, y, pc, cv):
	kf = KFold(n_splits=cv)
	rmse = []
	for train_index, test_index in kf.split(x):
		x_train, x_test = x[train_index], x[test_index]
		y_train, y_test = y[train_index], y[test_index]
		pls = PLSRegression(n_components=pc)
		pls.fit(x_train, y_train)
		y_predict = pls.predict(x_test)
		rmse.append(np.sqrt(mean_squared_error(y_test, y_predict)))
	rmse_mean = np.mean(rmse)
	return rmse_mean


def cars(x, y, fig_name, num=50, f=10, cv=5):
	'''
	CARS算法流程
	:param x: 光谱矩阵
	:param y: 理化值数据
	:param num: 采样次数
	:param f: 主成分数量
	:param cv: cv折交叉验证
	:return: 最佳波长的索引
	'''
	p = 0.8  # 选择80%数据集进入训练集
	m, n = x.shape
	u = np.power((n / 2), (1 / (num - 1)))
	k = (1 / (num - 1)) * np.log(n / 2)
	cal_num = np.round(m * p)  # 校正集数量
	# val_num = m - cal_num
	b2 = np.arange(n)  # 创建等差数列
	x = copy.deepcopy(x)  # 拷贝副本
	D = np.vstack((np.array(b2).reshape(1, -1), x))  # 垂直堆叠数据
	WaveData = []
	Coeff = []
	WaveNum = []
	RMSECV = []
	r = []
	for i in range(1, num + 1):
		r.append(u * np.exp(-1 * k * i))  # EDF筛选波长比例
		wave_num = int(np.round(r[i - 1] * n))  # 剩余波长数量
		WaveNum = np.hstack((WaveNum, wave_num))  # 水平方向上平铺
		cal_index = np.random.choice(np.arange(m), size=int(cal_num), replace=False)  # 对校正集打乱了顺序

		wave_index = b2[:wave_num].reshape(1, -1)[0]  # 波长的索引
		xcal = x[np.ix_(list(cal_index), list(wave_index))]  # 取得波长和样本的对应矩阵
		ycal = y[cal_index]  #
		x = x[:, wave_index]
		D = D[:, wave_index]
		d = D[0, :].reshape(1, -1)
		wnum = n - wave_num
		if wnum > 0:
			d = np.hstack((d, np.full((1, wnum), -1)))
		if len(WaveData) == 0:
			WaveData = d
		else:
			WaveData = np.vstack((WaveData, d.reshape(1, -1)))

		if wave_num < f:
			f = wave_num

		pls = PLSRegression(n_components=f)
		pls.fit(xcal, ycal)
		beta = pls.coef_  # 每个波长的回归系数
		b = np.abs(beta)
		b2 = np.argsort(-b, axis=0)  # 提取索引  由大到小排列
		coef = copy.deepcopy(beta)
		coeff = coef[b2, :].reshape(len(b2), -1)
		cb = coeff[:wave_num]

		if wnum > 0:
			cb = np.vstack((cb, np.full((wnum, 1), 0)))
		if len(Coeff) == 0:
			Coeff = copy.deepcopy(cb)
		else:
			Coeff = np.hstack((Coeff, cb))
		rmsecv, rindex = pc_cross(xcal, ycal, f, cv)
		RMSECV.append(cross(xcal, ycal, rindex + 1, cv))
	CoeffData = Coeff.T

	WAVE = []
	COEFF = []

	for i in range(WaveData.shape[0]):
		wd = WaveData[i, :]
		cd = CoeffData[i, :]
		WD = np.ones((len(wd)))
		CO = np.ones((len(wd)))
		for j in range(len(wd)):
			ind = np.where(wd == j)
			if len(ind[0]) == 0:
				WD[j] = 0
				CO[j] = 0
			else:
				WD[j] = wd[ind[0]]
				CO[j] = cd[ind[0]]
		if len(WAVE) == 0:
			WAVE = copy.deepcopy(WD)
		else:
			WAVE = np.vstack((WAVE, WD.reshape(1, -1)))
	if len(COEFF) == 0:
		COEFF = copy.deepcopy(CO)
	else:
		COEFF = np.vstack((WAVE, CO.reshape(1, -1)))

	MinIndex = np.argmin(RMSECV)
	minRMSECV = np.min(RMSECV)
	Optimal = WAVE[MinIndex, :]
	boindex = np.where(Optimal != 0)
	OptWave = boindex[0]

	fig = plt.figure()
	plt.rcParams['font.sans-serif'] = [u'Arial Unicode MS']
	fonts = 16
	plt.subplot(311)
	plt.ylabel('变量个数', fontsize=fonts)
	plt.title('最佳迭代次数：' + str(MinIndex) + '次', fontsize=fonts)
	plt.plot(np.arange(num), WaveNum)

	plt.subplot(312)
	plt.ylabel('误差', fontsize=fonts)
	plt.plot(np.arange(num), RMSECV)

	plt.subplot(313)
	plt.xlabel('蒙特卡洛迭代次数', fontsize=fonts)
	plt.ylabel('回归系数', fontsize=fonts)
	plt.plot(CoeffData)
	plt.vlines(MinIndex, CoeffData.min() - 1, CoeffData.max() + 1, colors='r')
	fig.savefig('{}.png'.format(fig_name))
	plt.close(fig)
	return OptWave, minRMSECV
