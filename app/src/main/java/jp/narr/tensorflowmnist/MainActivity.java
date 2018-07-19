/*
   Copyright 2016 Narrative Nights Inc. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

package jp.narr.tensorflowmnist;

import android.graphics.PointF;
import android.os.Bundle;
import android.speech.tts.TextToSpeech;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.widget.TextView;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;
import android.speech.tts.TextToSpeech;
import java.util.Locale;

public class MainActivity extends AppCompatActivity implements View.OnTouchListener{
	private static final String TAG = "MainActivity";
	static {
		System.loadLibrary("tensorflow_inference");
	}
	private TextToSpeech tts;
	private static final String model_file = "file:///android_asset/PBfile864.pb";

	//this is the graph with 512 units.
	private static final String input_node = "reshape_1_input";
	private static final long[] input_shape = {1,784};
	//private static final String output_node = "dense_2/Softmax";
	private static final String output_node = "dense_3/Softmax";
	TensorFlowInferenceInterface inferenceInterface;

	private static final int PIXEL_WIDTH = 28;

	private TextView mResultText;

	private float mLastX;
	private float mLastY;

	private DrawModel mModel;
	private DrawView mDrawView;

	private PointF mTmpPiont = new PointF();




	@SuppressWarnings("SuspiciousNameCombination")
	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_main);
		inferenceInterface = new TensorFlowInferenceInterface(getAssets(),model_file);
		tts = new TextToSpeech(MainActivity.this, new TextToSpeech.OnInitListener() {
			@Override
			public void onInit(int status) {
				if(status!=TextToSpeech.ERROR) {
					tts.setLanguage(Locale.US);
				}
			}
		});




		mModel = new DrawModel(PIXEL_WIDTH, PIXEL_WIDTH);

		mDrawView = (DrawView) findViewById(R.id.view_draw);
		mDrawView.setModel(mModel);
		mDrawView.setOnTouchListener(this);

		View detectButton = findViewById(R.id.button_detect);
		detectButton.setOnClickListener(new View.OnClickListener() {
			@Override
			public void onClick(View v) {
				onDetectClicked();
			}
		});

		View clearButton = findViewById(R.id.button_clear);
		clearButton.setOnClickListener(new View.OnClickListener() {
			@Override
			public void onClick(View v) {
				onClearClicked();
			}
		});

		mResultText = (TextView)findViewById(R.id.text_result);
	}

	@Override
	protected void onResume() {
		mDrawView.onResume();
		super.onResume();
	}

	@Override
	protected void onPause() {
		mDrawView.onPause();
		super.onPause();
	}

	@Override
	public boolean onTouch(View v, MotionEvent event) {
		int action = event.getAction() & MotionEvent.ACTION_MASK;

		if (action == MotionEvent.ACTION_DOWN) {
			processTouchDown(event);
			return true;

		} else if (action == MotionEvent.ACTION_MOVE) {
			processTouchMove(event);
			return true;

		} else if (action == MotionEvent.ACTION_UP) {
			processTouchUp();
			return true;
		}
		return false;
	}

	private void processTouchDown(MotionEvent event) {
		mLastX = event.getX();
		mLastY = event.getY();
		mDrawView.calcPos(mLastX, mLastY, mTmpPiont);
		float lastConvX = mTmpPiont.x;
		float lastConvY = mTmpPiont.y;
		mModel.startLine(lastConvX, lastConvY);
	}

	private void processTouchMove(MotionEvent event) {
		float x = event.getX();
		float y = event.getY();

		mDrawView.calcPos(x, y, mTmpPiont);
		float newConvX = mTmpPiont.x;
		float newConvY = mTmpPiont.y;
		mModel.addLineElem(newConvX, newConvY);

		mLastX = x;
		mLastY = y;
		mDrawView.invalidate();
	}

	private void processTouchUp() {
		mModel.endLine();
	}

	private void onDetectClicked() {
		float pixels[] = mDrawView.getPixelData();
		float[] result = fpre(pixels);
		display(result);


	}
	int index = 99;
	private void display(float[] result){
		String[] ans = {
				"0",
				"1",
				"2",
				"3",
				"4",
				"5",
				"6",
				"7",
				"8",
				"9",
				"A",
				"B",
				"C",
				"D",
				"E",
				"F",
				"G",
				"H",
				"I",
				"J",
				"K",
				"L",
				"M",
				"N",
				"O",
				"P",
				"Q",
				"R",
				"S",
				"T",
				"U",
				"V",
				"W",
				"X",
				"Y",
				"Z",
				"a",
				"b",
				"c",
				"d",
				"e",
				"f",
				"g",
				"h",
				"i",
				"j",
				"k",
				"l",
				"m",
				"n",
				"o",
				"p",
				"q",
				"r",
				"s",
				"t",
				"u",
				"v",
				"w",
				"x",
				"y",
				"z"
		};


		int mi = 0;
		float max = 0;
		for (int i =0;i<62;i++){
			if(result[i]>max){
				max = result[i];
				mi = i;
			}
			String mes = "Probability of "+i+": "+result[i];
			Log.d("mess",mes);
		}

		if(max>0.50f) {
			String resd = ans[mi];
			String con = String.format("%.1f", max * 100);
			String dt = "Detected = " + resd + " (" + con + "%)";
			mResultText.setText(dt);
			tts.speak(resd,TextToSpeech.QUEUE_FLUSH,null);
		}
		else{
			String resd = ans[mi];
			String con = String.format("%.1f", max * 100);
			String dt = "Maybe: " + resd + " (" + con + "%)";
			mResultText.setText(dt);
			tts.speak("Maybe " + resd+" .I am Not sure.",TextToSpeech.QUEUE_FLUSH,null);
		}
	}


	private float[] fpre(float[] pb){
	    //this is where the main ml work is happening. we are feeding the array image into the interface and recieving an array with 62 values.
        //all probabilities of the digit. run the app and check the log.
		inferenceInterface.feed(input_node,pb,input_shape);
		inferenceInterface.run(new String[] {output_node});
		float[] result = new float[62];
		float[] l = new float[620];
		inferenceInterface.fetch(output_node,result);
		return result;
	}

	private void onClearClicked() {
		mModel.clear();
		mDrawView.reset();
		mDrawView.invalidate();

		mResultText.setText("");
	}
}
