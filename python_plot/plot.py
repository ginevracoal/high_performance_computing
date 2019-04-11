import matplotlib
import os
matplotlib.use("Agg") # togliere bordo al grafico
import matplotlib.pyplot as plt

input_folder="times/"

def main():
	plot_by_dimension(1200)
	plot_by_dimension(12000)

def plot_by_dimension(size):

	# read input files from the given input folder
	input_files=os.listdir(input_folder)

	fig, axes=plt.subplots(1, 1, figsize=(16,9), dpi=200)

	# liste vuote
	mean_times = []
	xtics_labels = []

	# indice del file e nome del file in ordine di lettura
	for file_idx,file_name in enumerate(input_files):

		with open(os.path.join(input_folder,file_name)) as file_desc:
			file_text = file_desc.readlines()

		# pulisco il file 
		file_text_clean = [float(line.strip()) for line in file_text]

		# interpreto il nome del file con separatore _ 
		times_text, mode, matsize, numthreads = file_name.splitext("_")[0]

		if size!=int(matsize):
			continue	
		mean_times.append(sum(file_text_clean)/len(file_text_clean))

		# file_to_times[file_name]=file_text_clean

		# label the x axis
		xtics_labels.append(mode+" "+numthreads)

		# plot scatter points
		axes.scatter([file_idx]*len(file_text_clean), file_text_clean, alpha=0.3)

	axes.plot(range(len(input_files)), mean_times, 'b-')
	axes.grid()
	axes.set_xlabel("ciao")
	axes.set_ylabel("ciaociao")
	axes.set_title("ciaociggrao")
	axes.set_xticks()
	axes.set_xticklabels(xtics_labels)



	# print(file_to_times)




	# plt.show()

	plt.savefig("plot_{}.png".format(size))

	# 
	plt.clf()

# per importare
if __name__ == "__main__" : 
	main()