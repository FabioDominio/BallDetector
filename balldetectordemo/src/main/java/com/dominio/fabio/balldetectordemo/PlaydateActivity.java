package com.dominio.fabio.balldetectordemo;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.DialogInterface;
import android.content.Intent;
import android.media.MediaPlayer;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.StrictMode;
import android.util.Log;
import android.view.View;
import android.view.View.OnClickListener;
import android.widget.Button;
import android.widget.ImageView;

import org.opencv.samples.ballDetector.R;

/*
 * the main activity to host a plat date via WebRTC
 */
public class PlaydateActivity extends Activity {
    private static final String TAG = "PlaydateActivity";
	private Button danielButton = null;
	private Button charlieButton = null;
	private Button anneButton = null;
	private ImageView imageView = null;
	
	// state information
	private String chatRoom = null;
	public static String previousChatRoom = null;
	private boolean chatInProgress = false;

	// server related constants
	private static final String BASE_URL = "http://54.183.228.194:8080/SocialPlay-server/";
	private static final String SERVER_URL = BASE_URL + "socialPlay";
	private static final String GET_URL = SERVER_URL + "/1";
	private static final String UPDATE_URL = SERVER_URL + "/2";
	private static final String FIND_URL = SERVER_URL + "?action=findChatRoom";
	
	private MediaPlayer mp = null;
	private MediaPlayer mp2 = null;
	private PlayRingtoneTask pTask = null;
	private long duration = 0;
	private CheckExistingConnectionTask checkTask = null;
	
	public static String CHATROOM = "chatRoom";
	public static String IS_INITIATOR = "isInitiator";
	/**
	 * Called when the activity is first created. This is where we'll hook up
	 * our views in XML layout files to our application.
	 **/
	@Override
	public void onCreate(Bundle savedInstanceState) {
		disableStrictMode();
		super.onCreate(savedInstanceState);

		Log.e(TAG, "On create");
		setContentView(R.layout.playdate);

		// this is the button for a user to connect to a friend
		danielButton = (Button) findViewById(R.id.button1);
		danielButton.setOnClickListener(new OnClickListener() {
			@Override
			public void onClick(View v) {
				Log.e(TAG, "OnClick danielButton");

				danielButton.setText(R.string.connecting);
				disableButtons();
				findChatRoomAndConnect();
			}
		});

		charlieButton = (Button) findViewById(R.id.button2);
		charlieButton.setOnClickListener(new OnClickListener() {
			@Override
			public void onClick(View v) {
				Log.e(TAG, "charlieButton OnClick");

				danielButton.setText(R.string.connecting);
				disableButtons();
				findChatRoomAndConnect();
			}
		});

		anneButton = (Button) findViewById(R.id.button3);
		anneButton.setOnClickListener(new OnClickListener() {
			@Override
			public void onClick(View v) {
				Log.e(TAG, "charlieButton OnClick");

				anneButton.setText(R.string.connecting);
				disableButtons();
				findChatRoomAndConnect();
			}
		});
		
		imageView = (ImageView) findViewById(R.id.imageView1);
		imageView.setImageResource(R.drawable.invite);
		imageView.setVisibility(View.VISIBLE);
	}

	private void disableStrictMode() {
		if (android.os.Build.VERSION.SDK_INT > 9) {
			StrictMode.ThreadPolicy policy = new StrictMode.ThreadPolicy.Builder()
					.permitAll().build();
			StrictMode.setThreadPolicy(policy);
		}
	}
	
	private void disableButtons()
	{
		danielButton.setEnabled(false);
		charlieButton.setEnabled(false);
		anneButton.setEnabled(false);
	}
	@Override
	protected void onDestroy() {
		super.onDestroy();			
	}

	/**
	 * Called when the activity is coming to the foreground. This is where we
	 * will check whether there's an incoming connection.
	 **/
	@Override
	protected void onStart() {
		chatInProgress = false;
		Log.e(TAG, "onStart");
		super.onStart();
		

		checkTask = new CheckExistingConnectionTask();
		checkTask.execute();
		
		if(mp2 == null)
		{
			mp2 = MediaPlayer.create(this, R.raw.ringtone);
		}
		if(mp == null)
		{
			mp = MediaPlayer.create(this, R.raw.playdate);
		}
		duration = (long)mp.getDuration() + 500;
		mp.start();
		
		danielButton.setText(R.string.button_daniel);
		danielButton.setEnabled(true);
		
		charlieButton.setText(R.string.button_charlie);
		charlieButton.setEnabled(true);
		
		anneButton.setText(R.string.button_anne);
		anneButton.setEnabled(true);
	}
	
	/**
	 * utility to do sleep
	 * 
	 * @param milliseconds
	 */
	public static void sleep(long milliseconds) {
		try {
			Log.e(TAG, "will sleep " + milliseconds
					+ "milliseconds");
			Thread.sleep(milliseconds);
		} catch (Exception ex) {
			Log.e(TAG,
					"sleep encounters exception: " + ex.getMessage());
		}
	}

	/**
	 * the Async task to constantly poll whether there's an incoming connection
	 *
	 */
	private class CheckExistingConnectionTask extends
			AsyncTask<String, Void, String> {
		@Override
		protected String doInBackground(String... params) {
			String response = "";
			int count = 0;
			while (count < 1) {
				Log.e(TAG, "CheckExistingConnectionTask doInBackground chatInProgress: "
						+ chatInProgress);

				if (!chatInProgress) {
					Log.e(TAG,
							"getExistingConnection: will connnect to "
									+ GET_URL);
					// call to the server to find out whether there is
					// an incoming connection through an async task
					SocialPlayServer server = SocialPlayServer.get(GET_URL);
					response = server.performGet();
					if (response != null && !chatInProgress) {
						Log.e(TAG, "CheckExistingConnectionTask doInBackground response: "
								+ response + " and previousChatRoom=" + previousChatRoom);
						if(previousChatRoom == null || !response.equals(previousChatRoom))
							break;
					}
				}
				else
				{
					Log.e(TAG, "CheckExistingConnectionTask will break because chatInProgress: "
							+ chatInProgress);
					break;
				}
				sleep(1000);
			}
			return response;
		}
		
		@Override
		protected void onPostExecute(String result) {
			chatRoom = result;
			Log.e(TAG, "CheckExistingConnectionTask onPostExecute received: " + result);
			Log.e(TAG, "will start PlayRingtoneTask");
			pTask = new PlayRingtoneTask();
			pTask.execute();
			alertConnection("Your friend invites you to a play date",
					"Accept?");
		}
	}
	
	/**
	 * the Async task to constantly poll whether there's an incoming connection
	 *
	 */
	private class CheckConnectionEstablishedTask extends
			AsyncTask<Boolean, Void, Boolean> {
		@Override
		protected Boolean doInBackground(Boolean... params) {
			Boolean response = false;
			int count = 0;
			while (count < 10) {
				Log.e(TAG,
						"CheckConnectionEstablishedTask doInBackground count: "
								+ count);

				Log.e(TAG,
						"getExistingConnection: will connnect to " + GET_URL);
				// call to the server to find out whether there is
				// an incoming connection through an async task
				SocialPlayServer server = SocialPlayServer
						.isConnectionEstablished(GET_URL);
				if (server.performIsConnectionEstablished()) {
					response = true;
					break;
				}
				sleep(1000);
				count++;
			}
			return response;
		}
		
		@Override
		protected void onPostExecute(Boolean result) {
			Log.e(TAG, "onPostExecute result: "
					+ result);
			if(result)
			{
				// remote party accepted, now go to the video chat
				proceedToChat(true);
			}else{
				Log.e(TAG, "onPostExecute show alert and back to start");
				alert("Your friend didn't accept your invite", "Please try again!");
				onStart();
			}
		}
	}

	private void proceedToChat(boolean isInitiator)
	{
		imageView.setImageResource(R.drawable.treasurehunt);

		Intent intent = new Intent(getApplicationContext(), TreasureHuntImageActivity.class);
		intent.putExtra(CHATROOM, chatRoom);
		intent.putExtra(IS_INITIATOR, isInitiator ? "true" : "false");
		startActivity(intent);
	}
	/**
	 * the Async task to play ringtone upon an incoming connection
	 *
	 */
	private class PlayRingtoneTask extends
			AsyncTask<Void, Void, Void> {
		@Override
		protected Void doInBackground(Void... param) {
			int count = 0;
			
			while (count < 10) {
				Log.e(TAG, "PlayRingtoneTask doInBackground count: "
						+ count);

				if (!chatInProgress) {
					Log.e(TAG, "PlayRingtoneTask doInBackground start playing ringtone");
					mp2.start();
				}
				else
				{
					mp2.pause();
					break;
				}
				sleep(duration);
				count++;
			}
			
			return (Void)null;
		}
		
	}
	
	/**
	 * Show an alert dialog with message. user can either click OK or cancel no
	 * real action is triggered
	 * 
	 * @param header
	 * @param message
	 */
	private void alert(String header, String message) {
		AlertDialog.Builder builder = new AlertDialog.Builder(this);
		builder.setMessage(message)
				.setTitle(header)
				.setCancelable(true)
				.setPositiveButton(android.R.string.ok,
						new DialogInterface.OnClickListener() {
							@Override
							public void onClick(DialogInterface dialog,
									int which) {
								dialog.dismiss();
							}
						});
		AlertDialog alert = builder.create();
		alert.show();
	}

	/**
	 * pops up an alert dialog with 2 messages, one as the header and the other
	 * as the message to be shown in the dialog. if user clicks on NO, it enable
	 * the connect button on screen if user clicks on YES, it connects user to
	 * the incoming request
	 * 
	 * @param header
	 * @param message
	 */
	private void alertConnection(String header, String message) {
		AlertDialog.Builder builder = new AlertDialog.Builder(this);
		builder.setMessage(message)
				.setTitle(header)
				.setNegativeButton(R.string.no,
						new DialogInterface.OnClickListener() {
							@Override
							public void onClick(DialogInterface dialog,
									int which) {
								Log.e(TAG, "NO clicked, will stop mp2");
								stopRing();
								dialog.dismiss();
							}
						})
				.setPositiveButton(android.R.string.yes,
						new DialogInterface.OnClickListener() {
							@Override
							public void onClick(DialogInterface dialog,
									int which) {
								Log.e(TAG, "OK clicked, will stop mp2");
								stopRing();
								dialog.dismiss();
								
								// update the server that it accepted the request
								SocialPlayServer server = SocialPlayServer.updateConnection(UPDATE_URL);
								if(server.performIsConnectionEstablished()){
									Log.e(TAG, "connection established");
									proceedToChat(false);
								}
							}
						});
		AlertDialog alert = builder.create();
		alert.show();
	}
	
	private void stopRing()
	{
		if(mp2!=null){
			mp2.pause();
			chatInProgress = true;
		}

		if(pTask != null)
		{
			Log.e(TAG, "will cancel the pTask");
			pTask.cancel(true);
		}
	}
	/**
	 * method to find a chat room to join, this is for the initiator of a video
	 * chat
	 */
	private void findChatRoomAndConnect() {
		Log.e(TAG, "findChatRoomAndConnect");
		Log.e(TAG, "will cancel the checkTask");
		chatInProgress = true;
		if(checkTask != null)
			Log.e(TAG, "cancel the checkTask returns: " + checkTask.cancel(true));
		SocialPlayServer server = SocialPlayServer.findChatRoom(FIND_URL);
		server.fetchTo(new ServerFindListener());
	}
	
	/**
	 * Handle the result of an asynchronous Find request to server
	 **/
	private class ServerFindListener implements SocialPlayServer.Client {
		public void handleResponse(String chatRoomId, Exception error) {
			if (chatRoomId != null) {
				chatRoom = chatRoomId;
				Log.e(TAG, "ServerFindListener returns: "
						+ chatRoomId);
				createAsGet(SERVER_URL, chatRoomId);
			} else {
				Log.v(TAG, "Server error: " + error);
				// There was either a network error or server error.
				// Show alert for the latter.
				alert("Failed to find a chat room to join!",
						"Please try again!");
				onStart();
			}
		}
	}
	
	/**
	 * this is kind of hack for now to use a GET instead of a POST method to
	 * create an entry on the server side about the chat room currently joined
	 * 
	 * @param serverUrl
	 * @param chatRoomId
	 */
	private void createAsGet(String serverUrl, String chatRoomId) {
		Log.e(TAG, "create");
		SocialPlayServer server = SocialPlayServer.createAsGet(serverUrl,
				chatRoomId);
		server.getTo(new ServerCreateAsGetListener());
	}
	
	/**
	 * 
	 * Handle the result of an asynchronous GET request to server It's kind of
	 * hack now to use a GET instead of a CREATE
	 */
	private class ServerCreateAsGetListener implements SocialPlayServer.Client {
		public void handleResponse(String chatRoomId, Exception error) {
			Log.e(TAG, "ServerCreateAsGetListener returns: "
					+ chatRoomId);
			if (chatRoomId != null && error == null) {
				Log.e(TAG, "Sent To Server successfully");
				
				// record this chat room so it won't mistake as an incoming request
				previousChatRoom = chatRoomId;
				
				Log.e(TAG, "ServerCreateAsGetListener will start CheckConnectionEstablishedTask");
				// now waiting for the remote party to accept
				CheckConnectionEstablishedTask cTask = new CheckConnectionEstablishedTask();
				cTask.execute();
			} else {
				alert("Failed to communicate to server! Error: "
						+ error.getMessage(),
						"Other party can't join this chat!");
				
				onStart();
			}
		}
	}
}
