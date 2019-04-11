import matplotlib
import os

matplotlib.use("Agg") # togliere bordo al grafico
import matplotlib.pyplot as plt

input_folder="times/"

def main():

	plot_by_mode("weak")
	plot_by_mode("strong")

def plot_by_mode(scalingmode):

	# read input files from the given input folder
	input_files=os.listdir(input_folder)

	# a figure and 6 subplots sharing both axes
	fig, axes =plt.subplots(1, 6, figsize=(16,9), dpi=200)

	# liste vuote
	mean_times = []
	xtics_labels = []

# 	plot_by_dimension(10000000)
# 	plot_by_dimension(100000000)
# 	plot_by_dimension(1000000000)

# def plot_by_dimension(dim):

	# indice del file e nome del file in ordine di lettura
	for file_idx,file_name in enumerate(input_files):

		# leggo il file specificato dal percorso una riga per volta
		with open(os.path.join(input_folder,file_name)) as file_path:
			file_text = file_path.readlines()
		# elimino dalla lettura del testo \n
		file_text_clean = [float(line.strip()) for line in file_text]

		# interpreto il nome del file
		file_name_clean = file.splitext(".")[0]
		mode, scaling, matsize, numprocs = file.splitext("_")

		# calculate average execution time
		if scalingmode==mode:
			continue	
		mean_times.append(sum(file_text_clean)/len(file_text_clean))

		# file_to_times[file]=file_text_clean

		# label the x axis
		xtics_labels.append(numprocs)

		# plot scatter points
		axes.scatter([file_idx]*len(file_text_clean), file_text_clean, alpha=0.3)

		axes.plot(range(len(input_files)), mean_times, 'b-')
		axes.grid()
		# axes.set_xlabel("ciao")
		# axes.set_ylabel("ciaociao")
		# axes.set_title(" dimensione della matrice ")
		axes.set_xticks()
		axes.set_xticklabels(xtics_labels)

	# print(file_to_times)

	# plt.savefig("strong_scaling.png".format(size))
	plt.savefig(scalingmode + "_scaling.png")

	plt.clf()

	plt.show()

if __name__ == "__main__" : 
	main()