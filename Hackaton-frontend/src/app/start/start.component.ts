import { Component } from '@angular/core';
import { ApiService } from '../service/api.service';

@Component({
  selector: 'app-root',
  template: `
    <div>
      <input type="file" (change)="onFileSelected($event)" [accept]="'.json'">
      <button (click)="onUpload()">Upload</button>
    </div>
  `,
  styles: []
})
export class StartComponent {
  selectedFile: File | null = null;

  constructor(private apiService: ApiService) {}

  onFileSelected(event: any): void {
    this.selectedFile = event.target.files[0];
  }

  onUpload(): void {
    if (this.selectedFile) {
      this.apiService.uploadFile(this.selectedFile).subscribe(
        response => console.log('Upload success!', response),
        error => console.error('Error uploading file!', error)
      );
    } else {
      console.error('No file selected!');
    }
  }
}
