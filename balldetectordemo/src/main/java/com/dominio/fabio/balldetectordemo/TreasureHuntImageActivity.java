package com.dominio.fabio.balldetectordemo;

import android.app.Activity;
import android.content.Intent;
import android.media.MediaPlayer;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;

import org.opencv.samples.ballDetector.R;

/*
 * the main activity to host the webRTC connectivity
 */
public class TreasureHuntImageActivity extends Activity {
    private static final String TAG = "TreasureHuntImageActivity";
	private ImageView imageView = null;
	
	private MediaPlayer mp = null;
	private long duration = 0;

	/**
	 * Called when the activity is first created. This is where we'll hook up
	 * our views in XML layout files to our application.
	 **/
	@Override
	public void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		
		Log.e(TAG, "On create");
		setContentView(R.layout.treasurehunt);
		
		imageView = (ImageView) findViewById(R.id.treasurehuntImage);
		imageView.setVisibility(View.VISIBLE);
		imageView.setBackgroundResource(R.drawable.treasurehunt);
		imageView.bringToFront();
	}
	
	@Override
	protected void onDestroy() {
		if (null != mp) {
			mp.release();
		}
		super.onDestroy();
	}


	/**
	 * Called when the activity is coming to the foreground. This is where we
	 * will check whether there's an incoming connection.
	 **/
	@Override
	protected void onStart() {
		super.onStart();

		Log.e(TAG, "on start");
		
		if(mp == null)
		{
			mp = MediaPlayer.create(this, R.raw.treasurehunt);
			duration = mp.getDuration();
		}	
		mp.start();
		sleep(duration);
		
		Intent intent = getIntent();
	    
		String chatRoom = intent.getStringExtra(PlaydateActivity.CHATROOM);
		
		intent = new Intent(getApplicationContext(), GuidedPlayActivity.class);
		intent.putExtra(PlaydateActivity.CHATROOM, chatRoom);
		startActivity(intent);
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
