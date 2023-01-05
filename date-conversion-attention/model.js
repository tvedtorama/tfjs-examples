/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from '@tensorflow/tfjs';
import * as dateFormat from './date_format';

/**
 * A custom layer used to obtain the last time step of an RNN sequential
 * output.
 * What timestep, the output - or the state?  By timestep - is this calculation step?
 * How does this work with backpropagation and hardware tensors?
 */
class GetLastTimestepLayer extends tf.layers.Layer {
  constructor(config) {
    super(config || {});
    this.supportMasking = true;
  }

  computeOutputShape(inputShape) {
    const outputShape = inputShape.slice();
    outputShape.splice(outputShape.length - 2, 1);
    return outputShape;
  }

  call(input) {
    // Input is ATW 128, 12, 64.  The slice picks away all but the last value of the 12 "middle" dimensions, hence we end up with a [128,64] tensor.
    if (Array.isArray(input)) {
      input = input[0];
    }
    const inputRank = input.shape.length;
    tf.util.assert(inputRank === 3, `Invalid input rank: ${inputRank}`);
    // `gather` removes one dimension by selecting given indices. `squeeze` "flattens" by removing 1-length dimensions.  As there is only 1 length after gathering 1 index, this removes that dimension
    // Note: The 12 values are assumably the 12 characters, and the 64 values are the embeddings for each character.
    // Not sure about the 128 - this could be the batch size - but does this thing actually send in the full batch to the model?
    return input.gather([input.shape[1] - 1], 1).squeeze([1]);
  }

  static get className() {
    return 'GetLastTimestepLayer';
  }
}
tf.serialization.registerClass(GetLastTimestepLayer);

/**
 * Create an LSTM-based attention model for date conversion.
 *
 * @param {number} inputVocabSize Input vocabulary size. This includes
 *   the padding symbol. In the context of this model, "vocabulary" means
 *   the set of all unique characters that might appear in the input date
 *   string.
 * @param {number} outputVocabSize Output vocabulary size. This includes
 *   the padding and starting symbols. In the context of this model,
 *   "vocabulary" means the set of all unique characters that might appear in
 *   the output date string.
 * @param {number} inputLength Maximum input length (# of characters). Input
 *   sequences shorter than the length must be padded at the end.
 * @param {number} outputLength Output length (# of characters).
 * @return {tf.Model} A compiled model instance.
 */
export function createModel(
    inputVocabSize, outputVocabSize, inputLength, outputLength) {
  const embeddingDims = 64;
  const lstmUnits = 64;

  const encoderInput = tf.input({shape: [inputLength]});
  const decoderInput = tf.input({shape: [outputLength]});

  // First layer after input: Embeddings. Calculates embeddings for the letters of the input.
  let encoder = tf.layers.embedding({
    inputDim: inputVocabSize,
    outputDim: embeddingDims,
    inputLength,
    maskZero: true
  }).apply(encoderInput);  // Apply is actually very cool in that it can be used in imperative (eager) mode, to calculate values straight away - or here: to build a model for later execution.  This is done here, because input is a symbolic tensor.
  // Second layer takes the embedding into a lstm layer
  encoder = tf.layers.lstm({
    units: lstmUnits,
    returnSequences: true // This causes the output to contain all the states and one more dimension (of 12 entries). This extra dimensions in stripped by the GetLastTimestepLayer layer.
  }).apply(encoder);

  // Fetch the last value of the encoder, using the special slicer
  const encoderLast = new GetLastTimestepLayer({
    name: 'encoderLast'
  }).apply(encoder);

  // Second layer, the embedding
  let decoder = tf.layers.embedding({
    inputDim: outputVocabSize,
    outputDim: embeddingDims,
    inputLength: outputLength,
    maskZero: true
  }).apply(decoderInput);
  // Second layer is the lstm, where the final "state" of the encoder is the input.
  decoder = tf.layers.lstm({
    units: lstmUnits,
    returnSequences: true
  }).apply(decoder, {initialState: [encoderLast, encoderLast]}); // Note that encoderLast is a symbolic tensor, hence it's values will follow the process. Most likely not backpropagated through.

  // Attention is a "dot product" of the decoder and encoder.  The dot product does what it says, it computes the dot product of the various vectors, this is another vector, presumably with one-fewer dimensions.
  //    Note: the attention matrix is just a dot product and softmax, it does not hold weights or does not train in any way.
  //    Format: The attention is a nullx10x12 matrix after doing the dot product on nullx12x64 * nullx10x64
  let attention = tf.layers.dot({axes: [2, 2]}).apply([decoder, encoder]);
  attention = tf.layers.activation({
    activation: 'softmax',  // Softmax scales and balances the outputs so that they sum to 1
    name: 'attention'
  }).apply(attention);

  // And what is context? The "dot product" of the attention and the encoder? softmax(decoder dot encoder) dot encoder.
  //   This is a key operation!  Here the output steps of the encoder is mixed with the attention matrix to provide the actual raw data for mapping the output.
  //   The result is 10x12 * 10x64 = 10x64
  const context = tf.layers.dot({
    axes: [2, 1],
    name: 'context'
  }).apply([attention, encoder]);
  // Combined output (?) of context (softmax(decoder dot encoder) dot encoder) and decoder.  Both data sets are 10x64
  const decoderCombinedContext =
      tf.layers.concatenate().apply([context, decoder]);
  // A dense layer for each output of the `decoderCombinedContext` layer.  tanh activation, lstmUnits Wide
  let output = tf.layers.timeDistributed({
    layer: tf.layers.dense({
      units: lstmUnits,
      activation: 'tanh'
    })
  }).apply(decoderCombinedContext);
  // A dense layer is then applied to all the time steps (12) - allowing each of them to output WHAT?
  output = tf.layers.timeDistributed({
    layer: tf.layers.dense({
      units: outputVocabSize,
      activation: 'softmax'
    })
  }).apply(output);

  const model = tf.model({
    inputs: [encoderInput, decoderInput],  // This is how the two-input contract is defined.
    outputs: output
  });
  model.compile({
    loss: 'categoricalCrossentropy',  // CategoricalCrossentropy is used to compare probability distributions on one-hot vectors.
    optimizer: 'adam'
  });
  return model;
}

/**
 * Perform sequence-to-sequence decoding for date conversion.
 *
 * @param {tf.Model} model The model to be used for the sequence-to-sequence
 *   decoding, with two inputs:
 *   1. Encoder input of shape `[numExamples, inputLength]`
 *   2. Decoder input of shape `[numExamples, outputLength]`
 *   and one output:
 *   1. Decoder softmax probability output of shape
 *      `[numExamples, outputLength, outputVocabularySize]`
 * @param {string} inputStr Input date string to be converted.
 * @return {{outputStr: string, attention?: tf.Tensor}}
 *   - The `outputStr` field is the output date string.
 *   - If and only if `getAttention` is `true`, the `attention` field will
 *     be populated by attention matrix as a `tf.Tensor` of
 *     dtype `float32` and shape `[]`.
 */
export async function runSeq2SeqInference(
    model, inputStr, getAttention = false) {
  return tf.tidy(() => {
    const encoderInput = dateFormat.encodeInputDateStrings([inputStr]);
    const decoderInput = tf.buffer([1, dateFormat.OUTPUT_LENGTH]);
    decoderInput.set(dateFormat.START_CODE, 0, 0);

    for (let i = 1; i < dateFormat.OUTPUT_LENGTH; ++i) {
      const predictOut = model.predict(
          [encoderInput, decoderInput.toTensor()]); // Note: Predict call takes two inputs, the full source data and the incrementing decoder results
      // Fetch the index of the item with the highest value from the the third dimension (vocabulary)
      const output = predictOut.argMax(2).dataSync()[i - 1];
      predictOut.dispose();
      decoderInput.set(output, 0, i);
    }

    const output = {outputStr: ''};

    // The `tf.Model` instance used for the final time step varies depending on
    // whether the attention matrix is requested or not.
    let finalStepModel = model;
    if (getAttention) {
      // If the attention matrix is requested, construct a two-output model.
      // - The 1st output is the original decoder output.
      // - The 2nd output is the attention matrix.
      finalStepModel = tf.model({
        inputs: model.inputs,
        // This weird construct combines the current output with what might be the output of the attention layer
        outputs: model.outputs.concat([model.getLayer('attention').output])
      });
    }

    const finalPredictOut = finalStepModel.predict(
        [encoderInput, decoderInput.toTensor()]);
    let decoderFinalOutput;  // The decoder's final output.
    if (getAttention) {
      // The output is now a tuple, where the second entry is the attention matrix
      decoderFinalOutput = finalPredictOut[0];
      output.attention = finalPredictOut[1];
    } else {
      decoderFinalOutput = finalPredictOut;
    }

    // Redefine this from a tensor, with the highest pri - to the actual selected index - of the last character
    decoderFinalOutput =
      decoderFinalOutput.argMax(2).dataSync()[dateFormat.OUTPUT_LENGTH - 1];

    // Weird stuff: Add all the characters up to the last one, from the "decoder input" - which is most likely just the encoder output?
    for (let i = 1; i < decoderInput.shape[1]; ++i) {
      output.outputStr += dateFormat.OUTPUT_VOCAB[decoderInput.get(0, i)];
    }
    // Add the last character from the decoder output, still very weird.
    output.outputStr += dateFormat.OUTPUT_VOCAB[decoderFinalOutput];
    return output;
  });
}
