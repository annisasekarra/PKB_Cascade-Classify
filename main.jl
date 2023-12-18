using DataFrames, Serialization, Statistics
using Distances

binary_data = open("data_9m.mat", "r")
data = deserialize(binary_data)
close(binary_data)

if isa(data, Matrix)
    col_names = [:sepallength, :sepalwidth, :petallength, :petalwidth, :x5]
    data = DataFrame(convert(Matrix{Float64}, data), col_names)
end

class1 = data[data[!, :x5] .== 1.0, :]
class2 = data[data[!, :x5] .== 2.0, :]
class3 = data[data[!, :x5] .== 3.0, :]

means_class1 = [mean(skipmissing(class1[!, i])) for i in names(class1)[1:end-1]]
means_class2 = [mean(skipmissing(class2[!, i])) for i in names(class2)[1:end-1]]
means_class3 = [mean(skipmissing(class3[!, i])) for i in names(class3)[1:end-1]]

println("Means for Class 1:")
println(means_class1)

println("\nMeans for Class 2:")
println(means_class2)

println("\nMeans for Class 3:")
println(means_class3)

data_without_last_column = select(data, Not(:x5))

data_part1 = select(data_without_last_column, [:sepallength])
data_part2 = select(data_without_last_column, [:sepalwidth])
data_part3 = select(data_without_last_column, [:petallength])
data_part4 = select(data_without_last_column, [:petalwidth])

println("\nData Part 1:")
display(data_part1)

println("\nData Part 2:")
display(data_part2)

println("\nData Part 3:")
display(data_part3)

println("\nData Part 4:")
display(data_part4)

means_matrix = vcat(means_class1', means_class2', means_class3')
means_data = DataFrame(means_matrix, [:sepallength, :sepalwidth, :petallength, :petalwidth])

println("\nMeans Data:")
display(means_data)

means_data_part1 = select(means_data, [:sepallength])
means_data_part2 = select(means_data, [:sepalwidth])
means_data_part3 = select(means_data, [:petallength])
means_data_part4 = select(means_data, [:petalwidth])

println("\nMeans Data Part 1:")
display(means_data_part1)

println("\nMeans Data Part 2:")
display(means_data_part2)

println("\nMeans Data Part 3:")
display(means_data_part3)

println("\nMeans Data Part 4:")
display(means_data_part4)

euclidean_distance(v1, v2) = Euclidean()(v1, v2)

function predict_class(row, means_matrix, class_labels)
    distances = [euclidean_distance(row, mean_row) for mean_row in eachrow(means_matrix)]
    min_distance_index = argmin(distances)
    return class_labels[min_distance_index]
end

predicted_classes = DataFrame(:predicted_class => Vector{Float64}())

for i in 1:size(data_part1, 1)
    row = [data_part1[i, 1], data_part2[i, 1], data_part3[i, 1], data_part4[i, 1]]
    predicted_class = predict_class(row, means_matrix, [1.0, 2.0, 3.0])
    push!(predicted_classes, [predicted_class])
end

data_with_predictions = hcat(data_part1, data_part2, data_part3, data_part4, predicted_classes)

rename!(data_with_predictions, [:sepallength, :sepalwidth, :petallength, :petalwidth, :predicted_class])

println("\nData Part 1 to Part 4 with Predicted Classes:")
display(data_with_predictions)

function calculate_accuracy(predicted_classes, actual_classes)
    correct_predictions = sum(predicted_classes .== actual_classes)
    total_predictions = length(actual_classes)
    accuracy = correct_predictions / total_predictions
    return accuracy
end

actual_classes = data[!, :x5]
predicted_classes = convert(Vector{Float64}, data_with_predictions[!, :predicted_class])

accuracy_percentage = calculate_accuracy(predicted_classes, actual_classes) * 100

println("\nAccuracy: ", accuracy_percentage, "%")
