package com.dominio.fabio.balldetectordemo;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Intent;
import android.graphics.drawable.AnimationDrawable;
import android.hardware.Camera;
import android.media.MediaPlayer;
import android.os.Bundle;
import android.os.CountDownTimer;
import android.os.Environment;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.MotionEvent;
import android.view.SubMenu;
import android.view.SurfaceView;
import android.view.View;
import android.view.View.OnClickListener;
import android.view.View.OnTouchListener;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.opencv.samples.ballDetector.R;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;
import java.util.ListIterator;

/*
 * the main activity to host the webRTC connectivity
 */
public class GuidedPlayActivity extends Activity implements CvCameraViewListener2, OnTouchListener {
    private static final String TAG = "GuidedPlayActivity";
	private Button hangupButton = null;
	private ImageView imageView = null;
	private String chatRoom = null;
	private TextView scoreView = null;
	private TextView timerView = null;
    private static TextView savedTimerView = null;
	private static MediaPlayer mp = null;

    private static CountDownTimer countDownTimer = null;
    private final long startTime = 60 * 1000;
    private final long interval = 1 * 1000;
    private int challengeCount = 1;
    private static int score = 0;
    private static int challengeLimit = 2;

    private BallDetectorView mOpenCvCameraView;
    private List<Camera.Size> mResolutionList;
    private MenuItem[] mEffectMenuItems;
    private SubMenu mColorEffectsMenu;
    private MenuItem[] mResolutionMenuItems;
    private SubMenu mResolutionMenu;
    private BallDetector ballDetector;

    private final int RED = 1;
    private final int GREEN = 2;
    private final int YELLOW = 4;
    private final int BLUE = 3;

    private static int foundInstanceCount = 0;
    private static int foundWrongInstanceCount = 0;
    private final int FIRMCOUNT = 5;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                    mOpenCvCameraView.setOnTouchListener(GuidedPlayActivity.this);
                    ballDetector = new BallDetector();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    public GuidedPlayActivity() {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    public class MyCountDownTimer extends CountDownTimer {
        private boolean isCancelled = false;
        public MyCountDownTimer(long startTime, long interval) {
            super(startTime, interval);
        }

        @Override
        public void onFinish() {
        	challengeCount++;
			if (challengeCount <= challengeLimit) {
				playAudio();
				startTimer(startTime, interval);
			}
			else
			{
                savedTimerView.setText("");
//                savedTimerView.setVisibility(View.INVISIBLE);
			}
        }

        @Override
        public void onTick(long millisUntilFinished) {
            if (!isCancelled) {
                Log.e(TAG, "onTick: millisUntilFinished = " + millisUntilFinished);
                savedTimerView.setText(R.string.remaining);
                savedTimerView.setText(savedTimerView.getText() + "" + millisUntilFinished / 1000);
                savedTimerView.setVisibility(View.VISIBLE);

                if (foundInstanceCount >= FIRMCOUNT) {
                    wonChallenge();

                    isCancelled = true;
                    this.cancel();
                    // force the current challenge to finish
                    onFinish();
                }

                if (foundWrongInstanceCount >= FIRMCOUNT) {
                    foundWrongInstanceCount = 0;
                    if (!mp.isPlaying())
                        playWrongColorInstance();
                }
            }
        }
    }

    private void playWrongColorInstance()
    {
        mp = MediaPlayer.create(this, R.raw.wrongcolor);
        long duration = (long)mp.getDuration() + 500;
        mp.start();
    }

    private void wonChallenge()
    {
        foundInstanceCount = 0;
//        String caption = "You found it! Win 10 Points!";
//        Toast.makeText(this, caption, Toast.LENGTH_SHORT).show();
        mp = MediaPlayer.create(this, R.raw.youwon);
        long duration = (long)mp.getDuration() + 500;
        mp.start();

        score += 10;
        displayScore();

//        if(countDownTimer != null)
//        {
//            countDownTimer.cancel();
//            savedTimerView.setText("");
//            countDownTimer = null;
//        }
        sleep(duration);
    }

    private void displayScore()
    {
		scoreView.setText(R.string.score);		
		scoreView.setText(scoreView.getText() + "" + score);    	
    }
    
	/**
	 * Called when the activity is first created. This is where we'll hook up
	 * our views in XML layout files to our application.
	 **/
	@Override
	public void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);

		// show a progress bar
		getWindow().requestFeature(Window.FEATURE_PROGRESS);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

		Log.e(TAG, "On create : " + (countDownTimer == null));
		setContentView(R.layout.guidedplay);

        mOpenCvCameraView = (BallDetectorView) findViewById(R.id.balldetectorview);

        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);

        mOpenCvCameraView.setCvCameraViewListener(this);

		hangupButton = (Button) findViewById(R.id.hangup_button);
		hangupButton.setOnClickListener(new OnClickListener() {
			@Override
			public void onClick(View v) {
				Log.e(TAG, "hangupButton OnClick");

				// when clicked, it hang up the call and clean the webview
				clearView();
			}
		});


		scoreView = (TextView) findViewById(R.id.score);
		scoreView.setText(R.string.score);
		displayScore();
		timerView = (TextView) findViewById(R.id.timer);
        savedTimerView = timerView;

//		imageView = (ImageView) findViewById(R.id.guidedImageView);
//        imageView.setImageResource(R.drawable.puff_cow);
//        imageView.setVisibility(View.VISIBLE);
//		imageView.setBackgroundResource(R.drawable.start);
//		imageView.post(new Runnable() {
//			@Override
//			public void run() {
//				AnimationDrawable frameAnimation = (AnimationDrawable) imageView
//						.getBackground();
//				frameAnimation.start();
//			}
//		});
	}

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_3, this, mLoaderCallback);
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
    }

    public void onCameraViewStopped() {
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        Mat frame = inputFrame.rgba();
        int result = ballDetector.findBall(frame);
//        Log.e(TAG, "onCameraFrame returns result: " + result);
        showResult(result);
        return frame;
    }

    private void checkResult(int color)
    {
        switch (challengeCount)
        {
            case 1:
                if(color == RED) foundInstanceCount++;
                else foundWrongInstanceCount++;
                break;
            case 2:
                if(color == YELLOW) foundInstanceCount++;
                else foundWrongInstanceCount++;
                break;
            default:
                foundWrongInstanceCount++;
                break;
        }

    }
    private void showResult(int result)
    {
        int x = result & 0x10;
        if( x == 16) {
            Log.e(TAG, "found ball shape!");
            int color = result & 0xf;

            checkResult(color);

            switch (color) {
                case RED:
                    Log.e(TAG, "it's red!");
                    break;
                case GREEN:
                    Log.e(TAG, "it's green!");
                    break;
                case BLUE:
                    Log.e(TAG, "it's blue!");
                    break;
                case YELLOW:
                    Log.e(TAG, "it's yellow!");
                    break;
            }
        }
        else
        {
//            Log.e(TAG, "didn't find ball shape!");
        }
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        int idx = 0;
        List<String> effects = mOpenCvCameraView.getEffectList();

        if (effects == null) {
            Log.e(TAG, "Color effects are not supported by device!");
            //return true;
        }
        else {
            mColorEffectsMenu = menu.addSubMenu("Color Effect");
            mEffectMenuItems = new MenuItem[effects.size()];


            ListIterator<String> effectItr = effects.listIterator();
            while (effectItr.hasNext()) {
                String element = effectItr.next();
                mEffectMenuItems[idx] = mColorEffectsMenu.add(1, idx, Menu.NONE, element);
                idx++;
            }
        }
        mResolutionMenu = menu.addSubMenu("Resolution");
        mResolutionList = mOpenCvCameraView.getResolutionList();
        mResolutionMenuItems = new MenuItem[mResolutionList.size()];

        ListIterator<Camera.Size> resolutionItr = mResolutionList.listIterator();
        idx = 0;
        while(resolutionItr.hasNext()) {
            Camera.Size element = resolutionItr.next();
            mResolutionMenuItems[idx] = mResolutionMenu.add(2, idx, Menu.NONE,
                    Integer.valueOf(element.width).toString() + "x" + Integer.valueOf(element.height).toString());
            idx++;
        }

        return true;
    }

    public boolean onOptionsItemSelected(MenuItem item) {
        Log.i(TAG, "called onOptionsItemSelected; selected item: " + item);
        if (item.getGroupId() == 1)
        {
            mOpenCvCameraView.setEffect((String) item.getTitle());
            Toast.makeText(this, mOpenCvCameraView.getEffect(), Toast.LENGTH_SHORT).show();
        }
        else if (item.getGroupId() == 2)
        {
            int id = item.getItemId();
            Camera.Size resolution = mResolutionList.get(id);
            mOpenCvCameraView.setResolution(resolution);
            resolution = mOpenCvCameraView.getResolution();
            String caption = Integer.valueOf(resolution.width).toString() + "x" + Integer.valueOf(resolution.height).toString();
            Toast.makeText(this, caption, Toast.LENGTH_SHORT).show();
        }

        return true;
    }

    @SuppressLint("SimpleDateFormat")
    @Override
    public boolean onTouch(View v, MotionEvent event) {
        Log.i(TAG,"onTouch event");
        SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd_HH-mm-ss");
        String currentDateandTime = sdf.format(new Date());
        String fileName = Environment.getExternalStorageDirectory().getPath() +
                "/sample_picture_" + currentDateandTime + ".jpg";
        mOpenCvCameraView.takePicture(fileName);
        Toast.makeText(this, fileName + " saved", Toast.LENGTH_SHORT).show();
        return false;
    }

    private void playAudio()
	{
		mp = MediaPlayer.create(this, (challengeCount == 1) ? R.raw.challenge1
				: R.raw.challenge2);

//		long duration = (long) mp.getDuration() + 500;
		mp.start();
//		PlaydateActivity.sleep(duration);	
	}
	
	private void startTimer(long period, long interval)
	{
        foundInstanceCount = 0;
        foundWrongInstanceCount = 0;
        countDownTimer = new MyCountDownTimer(period, interval);
        countDownTimer.start();
	}

	/**
	 * @Title: clearView
	 * @Description: Hang up method , clear webview cache
	 * @param
	 * @return void
	 * @throws
	 */
	protected void clearView() {
		Log.e(TAG, "clearView will clear view");

		hangupButton.setEnabled(false);

		PlaydateActivity.previousChatRoom = chatRoom; // remember the previous chat room
		Log.e(TAG,
				"clearView preserve previous chat room: "
						+ PlaydateActivity.previousChatRoom);

        // stop the timer activity
        if(countDownTimer != null)
        {
            countDownTimer.cancel();
            timerView.setText("");
            countDownTimer = null;
        }

        if(mp != null)
        {
            mp.pause();
            mp = null;
        }

		startActivity(new Intent(getApplicationContext(),
				PlaydateActivity.class));
	}

	/**
	 * Called when the activity is coming to the foreground. This is where we
	 * will check whether there's an incoming connection.
	 **/
	@Override
	protected void onStart() {
		super.onStart();

		Log.e(TAG, "on start: chatRoom = " + chatRoom);

        if(countDownTimer == null){
            playAudio();
            startTimer(startTime, interval);
        }
	}

	/**
	 * utility to do sleep
	 * 
	 * @param milliseconds
	 */
	public void sleep(long milliseconds) {
		try {
			Log.e(TAG, "will sleep " + milliseconds
					+ "milliseconds");
			Thread.sleep(milliseconds);
		} catch (Exception ex) {
			Log.e(TAG,
					"sleep encounters exception: " + ex.getMessage());
		}
	}

}
