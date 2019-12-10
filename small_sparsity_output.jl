
using CSV, DataFrames
ad = CSV.read("ML_project_data/ad.csv", header=false, categorical=true)
aps = CSV.read("ML_project_data/aps.data", header=false, categorical=true)
census = CSV.read("ML_project_data/census.csv", categorical=true)
# covertype = CSV.read("ML_project_data/covertype.data", header=false, categorical=true)
heart = CSV.read("ML_project_data/heart_disease.data", header=false, categorical=true, missingstring="?")

for df in [ad, aps, census, heart]
    for i in 1:size(df,2)
        if typeof(df[:,i]) == Array{Union{Missing, String},1}
            df[:,i] = CategoricalArray(df[:,i])
        elseif typeof(df[:,i]) == Array{String,1}
            df[:,i] = CategoricalArray(df[:,i])
        end
    end
end

let i = 0
	global counter() = (i += 1)
end;

dfnames = ["ad", "aps", "census", "heart"]
for df in [ad, aps, census, heart]
    i=counter()
    println(i)
    
    numFeatures = size(df, 2) - 1
    
    if numFeatures <= 15
    	kSet = collect(1:numFeatures)
    else
    	kSet = collect(5:5:20)
        push!(kSet, Int(round( 0.25 * numFeatures)))
    	push!(kSet, Int(round( 0.35 * numFeatures)))
    	push!(kSet, Int(round( 0.50 * numFeatures)))
    	push!(kSet, Int(round( 0.65 * numFeatures)))
    	push!(kSet, Int(round( 0.80 * numFeatures)))
    end
    	
	output = zeros(length(kSet), 4, 5)
	
	    
	# Split data into features and outcomes
	X = df[:,1:size(df,2)-1]
	y = df[:,size(df,2)]
	
	for k in 1:5
		
		# Split into training/validation/testing
		(train_X, train_y), (test_X, test_y) = IAI.split_data(:classification, X, y, train_proportion = 0.7);
	    	
	    for j in 1:length(kSet)
		
		    # define feature selection classifier
		    FSgrid = IAI.GridSearch(
		        IAI.OptimalFeatureSelectionClassifier(
		        ),
		        sparsity = kSet[j],
		    )
	        
	        IAI.fit!(FSgrid, train_X, train_y, validation_criterion=:auc);
		
		    ###############################################################################################################
		    ## This is the block of code that pulls the list of features from the feature selection model (called "sparse")
		    f = IAI.get_prediction_weights(FSgrid)
		    sparse = append!(collect(keys(f[1])), collect(keys(f[2])))
		    # println("Features selected for sparsity ",kSet[j], " for ",  dfnames[i], ": ", sparse)
		    ##
		    ###############################################################################################################
		
		    # redefine X to only include the features we've selected
		    train_X_sparse = train_X[sparse]
		    test_X_sparse = test_X[sparse]
		
		    # fit an optimal tree classifier based on selected features
		    lnr = IAI.OptimalTreeClassifier()
		    lnr.criterion = :entropy;
		    grid = IAI.GridSearch(lnr,
		        max_depth=[4, 7, 10],
		        minbucket=5
		    )
			
			output[j, 1, k] = j  # sparsity
		    output[j, 2, k] = @elapsed IAI.fit!(grid, train_X_sparse, train_y, validation_criterion=:auc) # runtime
		    output[j, 3, k] = IAI.score(grid, train_X_sparse, train_y, criterion=:entropy) # in sample entropy
		    output[j, 4, k] = IAI.score(grid, test_X_sparse, test_y, criterion=:auc) # out of sample AUC
		
		    # write the tree visualization to a file
		    IAI.write_html("ML_project_data/trees/" * dfnames[i] * string(j) * "Tree.html", IAI.get_learner(grid))
		        
		end
	
	end
	
	println("\n", output, "\n")
    
end

