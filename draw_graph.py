import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class PageRankComaprison():
	def __init__(self):
		# f = 0.95, prob_cutoff = 0.01, alpha = 0.3
		self._data = {
			'sample_size'	: [100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500],
			'o_time'		: [1490,2638,3984,5928,7155,8786,9729,10975,12349,14358,15383,16757,18613,19586,20611],
			'm_time'		: [1283,1947,2098,3057,3449,4103,4858,5038,5845,6422,6542,6987,7948,8143,8446],
			'error'			: [11.73,11.65,12.5,12.2,12.3,11.8,12.4,12.1,12.2,12.1,12.0,11.9,12.0,12.2,12.0]
		}

		# varying f value
		# sample_size = 1000, prob_cutoff = 0.01, alpha = 0.3
		self._fdata = {
			'f' 			: [0.5,0.6,0.7,0.8,0.9],
			'time'			: [76,3619,5623,6234,6303],
			'error'			: [22.3,18.7,13.7,12.4,11.9]
		}
	def draw(self):
		# Draw comparison between the two algorithms with varying sample size
		plt.subplot(2,1,1)
		plt.plot(self._data['sample_size'], self._data['o_time'], c='red')
		plt.plot(self._data['sample_size'], self._data['m_time'], c='blue')
		plt.xlabel('Sample Size')
		plt.ylabel('Time (ms)')
		red_patch = mpatches.Patch(color='red', label='Original Algorithm')
		blue_patch = mpatches.Patch(color='blue', label='Modified Algorithm')
		plt.legend(handles=[red_patch, blue_patch])

		plt.subplot(2,1,2)
		plt.plot(self._data['sample_size'], self._data['error'], c='red')
		plt.xlabel('Sample Size')
		plt.ylabel('Difference (%)')

		plt.savefig('page-rank-comparison-sample.png')
		plt.clf()

	def fdraw(self):
		# Draw comparison between the two algorithms with varying sample size
		plt.subplot(2,1,1)
		plt.plot(self._fdata['f'], self._fdata['time'], c='red')
		plt.xlabel('f value')
		plt.ylabel('Time (ms)')

		plt.subplot(2,1,2)
		plt.plot(self._fdata['f'], self._fdata['error'], c='red')
		plt.xlabel('f value')
		plt.ylabel('Difference (%)')

		plt.savefig('page-rank-comparison-f.png')
		plt.clf()

if __name__ == '__main__':
	pr = PageRankComaprison()
	pr.draw()
	pr.fdraw()
