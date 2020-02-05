package com.example.tflitetry;

import android.app.Activity;
import android.content.Intent;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;
import java.util.Objects;


public class MainActivity extends AppCompatActivity {

    ImageView imageView;
    TextView result_tv;
    Button btnCamera, btnGallery, btnClassify;
    private static final int PICK_IMAGE = 1;
    Uri imageUri;
    Bitmap bitmap;
    private static final String modelFile = "converted.tflite";
    private Interpreter tflite;
    ByteBuffer buffer;

    /** fashion_model.tflite and converted.tflite are both the same.
     * Only one difference, converted.tflite has slightly more accuracy.
     */

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        btnCamera = findViewById(R.id.btnCamera);
        imageView = findViewById(R.id.imageView);
        btnGallery = findViewById(R.id.btnGallery);
        btnClassify = findViewById(R.id.btnClassify);
        result_tv = findViewById(R.id.result_tv);

    }

    public void btnClassifyClicked(View v){
        float[][] out = new float[1][10];
        int bytes = bitmap.getByteCount();
        buffer = ByteBuffer.allocate(bytes);
        bitmap.copyPixelsToBuffer(buffer);

        try {
            tflite = new Interpreter(loadModelFile(this, modelFile));
        } catch (IOException e) {
            e.printStackTrace();
        }


        tflite.run(buffer, out);

        for(int i = 0; i < 10; i++) {
            Log.e("number", String.valueOf(out[0][i]));
            if (out[0][i] <= 0.5) {
                out[0][i] = 0;
            } else {
                out[0][i] = 1;
            }
        }

        result_tv.setText(Arrays.deepToString(out));
    }

    public void btnGalleryClicked(View v){
        Intent intent = new Intent();
        intent.setType("image/*");
        intent.setAction(Intent.ACTION_GET_CONTENT);

        startActivityForResult(Intent.createChooser(intent, "Select Picture"), PICK_IMAGE);
    }

    public void btnCameraClicked(View v){
        Intent intent = new Intent();
        intent.setType("image/*");
        intent.setAction(Intent.ACTION_GET_CONTENT);

        startActivityForResult(Intent.createChooser(intent, "Select Picture"), PICK_IMAGE);
    }


    private MappedByteBuffer loadModelFile(Activity activity, String MODEL_FILE) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(MODEL_FILE);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }


    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);


        if (requestCode==0 && resultCode == RESULT_OK){
            assert data != null;
            bitmap = (Bitmap) Objects.requireNonNull(data.getExtras()).get("data");
            imageView.setImageBitmap(bitmap);
        }

        if (requestCode==PICK_IMAGE && resultCode == RESULT_OK){
            assert data != null;
            imageUri = data.getData();
            try{
                bitmap = MediaStore.Images.Media.getBitmap(getContentResolver(),imageUri);
                imageView.setImageBitmap(bitmap);
            }
            catch (IOException e) {
                e.printStackTrace();
            }
        }
    }


}
